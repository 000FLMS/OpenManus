'use server';

import { AuthWrapperContext, withUserAuth } from '@/lib/auth-wrapper';
import { decryptLongTextWithPrivateKey, decryptWithPrivateKey } from '@/lib/crypto';
import { LANGUAGE_CODES } from '@/lib/language';
import { prisma } from '@/lib/prisma';
import { to } from '@/lib/to';
import fs from 'fs';
import path from 'path';
import fsPromises from 'fs/promises';

const MANUS_URL = process.env.MANUS_URL || 'http://localhost:5172';

const privateKey = fs.readFileSync(path.join(process.cwd(), 'keys', 'private.pem'), 'utf8');

export const getTask = withUserAuth(async ({ organization, args }: AuthWrapperContext<{ taskId: string }>) => {
  const { taskId } = args;
  const task = await prisma.tasks.findUnique({
    where: { id: taskId, organizationId: organization.id },
    include: { progresses: { orderBy: { index: 'asc' } } },
  });
  return task;
});

export const pageTasks = withUserAuth(async ({ organization, args }: AuthWrapperContext<{ page: number; pageSize: number }>) => {
  const { page = 1, pageSize = 10 } = args || {};
  const tasks = await prisma.tasks.findMany({
    where: { organizationId: organization.id },
    skip: (page - 1) * pageSize,
    take: pageSize,
    orderBy: { createdAt: 'desc' },
  });
  const total = await prisma.tasks.count();
  return { tasks, total };
});

type CreateTaskArgs = {
  modelId: string;
  prompt: string;
  tools: string[];
  files: File[];
  shouldPlan: boolean;
};
export const createTask = withUserAuth(async ({ organization, args }: AuthWrapperContext<CreateTaskArgs>) => {
  const { modelId, prompt, tools, files, shouldPlan } = args;
  console.log('createTask', args);
  console.log('organization', organization);
  const llmConfig = await prisma.llmConfigs.findUnique({ where: { id: modelId, organizationId: organization.id } });

  if (!llmConfig) throw new Error('LLM config not found');

  const preferences = await prisma.preferences.findUnique({
    where: { organizationId: organization.id },
  });

  // Query tool configurations
  const organizationTools = await prisma.organizationTools.findMany({
    where: { organizationId: organization.id, tool: { id: { in: tools } } },
    include: { tool: true },
  });

  // Build tool list, use configuration if available, otherwise use tool name
  const processedTools = tools.map(tool => {
    const orgTool = organizationTools.find(ot => ot.tool.id === tool);
    if (orgTool) {
      const env = orgTool.env ? JSON.parse(decryptLongTextWithPrivateKey(orgTool.env, privateKey)) : {};
      const query = orgTool.query ? JSON.parse(decryptLongTextWithPrivateKey(orgTool.query, privateKey)) : {};
      const fullUrl = buildMcpSseFullUrl(orgTool.tool.url, query);
      const headers = orgTool.headers ? JSON.parse(decryptLongTextWithPrivateKey(orgTool.headers, privateKey)) : {};

      return JSON.stringify({
        id: orgTool.tool.id,
        name: orgTool.tool.name,
        command: orgTool.tool.command,
        args: orgTool.tool.args,
        env: env,
        url: fullUrl,
        headers: headers,
      });
    }
    return tool;
  });

  console.log('Processed Tools:', processedTools);

  // Create task
  const task = await prisma.tasks.create({
    data: {
      prompt,
      status: 'pending',
      llmId: llmConfig.id,
      organizationId: organization.id,
      tools,
    },
  });

  const formData = new FormData();
  formData.append('task_id', `${organization.id}/${task.id}`);
  formData.append('prompt', prompt);
  formData.append('should_plan', shouldPlan.toString());
  processedTools.forEach(tool => formData.append('tools', tool));
  formData.append('preferences', JSON.stringify({ language: LANGUAGE_CODES[preferences?.language as keyof typeof LANGUAGE_CODES] }));
  formData.append(
    'llm_config',
    JSON.stringify({
      model: llmConfig.model,
      base_url: llmConfig.baseUrl,
      api_key: decryptWithPrivateKey(llmConfig.apiKey, privateKey),
      max_tokens: llmConfig.maxTokens,
      max_input_tokens: llmConfig.maxInputTokens,
      temperature: llmConfig.temperature,
      api_type: llmConfig.apiType || '',
      api_version: llmConfig.apiVersion || '',
    }),
  );
  files.forEach(file => formData.append('files', file, file.name));

  const [error, response] = await to(
    fetch(`${MANUS_URL}/tasks`, {
      method: 'POST',
      body: formData,
    }).then(async res => {
      if (res.status === 200) {
        return (await res.json()) as Promise<{ task_id: string }>;
      }
      throw Error(`Server Error: ${JSON.stringify(await res.json())}`);
    }),
  );

  if (error || !response.task_id) {
    await prisma.tasks.update({ where: { id: task.id }, data: { status: 'failed' } });
    throw error || new Error('Unkown Error');
  }

  await prisma.tasks.update({ where: { id: task.id }, data: { outId: response.task_id, status: 'processing' } });

  // Handle event stream in background
  handleTaskEvents(task.id, response.task_id, organization.id).catch(error => {
    console.error('Failed to handle task events:', error);
  });

  return { id: task.id, outId: response.task_id };
});

export const restartTask = withUserAuth(
  async ({
    organization,
    args,
  }: AuthWrapperContext<{ taskId: string; modelId: string; prompt: string; tools: string[]; files: File[]; shouldPlan: boolean }>) => {
    const { taskId, modelId, prompt, tools, files, shouldPlan } = args;

    const llmConfig = await prisma.llmConfigs.findUnique({ where: { id: modelId, organizationId: organization.id } });

    if (!llmConfig) throw new Error('LLM config not found');

    const preferences = await prisma.preferences.findUnique({
      where: { organizationId: organization.id },
    });

    // Query tool configurations
    const organizationTools = await prisma.organizationTools.findMany({
      where: { organizationId: organization.id, tool: { id: { in: tools } } },
      include: { tool: true },
    });

    // Build tool list, use configuration if available, otherwise use tool name
    const processedTools = tools.map(tool => {
      const orgTool = organizationTools.find(ot => ot.tool.id === tool);
      if (orgTool) {
        const env = orgTool.env ? JSON.parse(decryptLongTextWithPrivateKey(orgTool.env, privateKey)) : {};
        const query = orgTool.query ? JSON.parse(decryptLongTextWithPrivateKey(orgTool.query, privateKey)) : {};
        const fullUrl = buildMcpSseFullUrl(orgTool.tool.url, query);
        const headers = orgTool.headers ? JSON.parse(decryptLongTextWithPrivateKey(orgTool.headers, privateKey)) : {};

        return JSON.stringify({
          id: orgTool.tool.id,
          name: orgTool.tool.name,
          command: orgTool.tool.command,
          args: orgTool.tool.args,
          env: env,
          url: fullUrl,
          headers: headers,
        });
      }
      return tool;
    });

    const task = await prisma.tasks.findUnique({ where: { id: taskId, organizationId: organization.id } });
    if (!task) throw new Error('Task not found');
    if (task.status !== 'completed' && task.status !== 'terminated' && task.status !== 'failed') throw new Error('Task is processing');

    const progresses = await prisma.taskProgresses.findMany({
      where: { taskId: task.id, type: { in: ['agent:lifecycle:start', 'agent:lifecycle:complete'] } },
      select: { type: true, content: true },
      orderBy: { index: 'asc' },
    });

    const history = progresses.reduce(
      (acc, progress) => {
        if (progress.type === 'agent:lifecycle:start') {
          acc.push({ role: 'user', message: (progress.content as { request: string }).request });
        } else if (progress.type === 'agent:lifecycle:complete') {
          const latestUserProgress = acc.findLast(item => item.role === 'user');
          if (latestUserProgress) {
            acc.push({ role: 'assistant', message: (progress.content as { results: string[] }).results.join('\n') });
          }
        }
        return acc;
      },
      [] as { role: string; message: string }[],
    );

    // Send task to API
    const formData = new FormData();
    formData.append('task_id', `${organization.id}/${task.id}`);
    formData.append('prompt', prompt);
    formData.append('should_plan', shouldPlan.toString());
    processedTools.forEach(tool => formData.append('tools', tool));
    formData.append('preferences', JSON.stringify({ language: LANGUAGE_CODES[preferences?.language as keyof typeof LANGUAGE_CODES] }));
    formData.append(
      'llm_config',
      JSON.stringify({
        model: llmConfig.model,
        base_url: llmConfig.baseUrl,
        api_key: decryptWithPrivateKey(llmConfig.apiKey, privateKey),
        max_tokens: llmConfig.maxTokens,
        max_input_tokens: llmConfig.maxInputTokens,
        temperature: llmConfig.temperature,
        api_type: llmConfig.apiType || '',
        api_version: llmConfig.apiVersion || '',
      }),
    );
    formData.append('history', JSON.stringify(history));
    files.forEach(file => formData.append('files', file));
    console.log(formData);
    const [error, response] = await to(
      fetch(`${MANUS_URL}/tasks/restart`, {
        method: 'POST',
        body: formData,
      }).then(res => res.json() as Promise<{ task_id: string }>),
    );

    if (error || !response.task_id) {
      throw new Error('Failed to restart task');
    }

    await prisma.tasks.update({ where: { id: task.id }, data: { outId: response.task_id, status: 'processing' } });

    // Handle event stream in background
    handleTaskEvents(task.id, response.task_id, organization.id).catch(error => {
      console.error('Failed to handle task events:', error);
    });

    return { id: task.id, outId: response.task_id };
  },
);

export const terminateTask = withUserAuth(async ({ organization, args }: AuthWrapperContext<{ taskId: string }>) => {
  const { taskId } = args;

  const task = await prisma.tasks.findUnique({ where: { id: taskId, organizationId: organization.id } });
  if (!task) throw new Error('Task not found');
  if (task.status !== 'processing' && task.status !== 'terminating') {
    return;
  }

  const [error] = await to(
    fetch(`${MANUS_URL}/tasks/terminate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task_id: `${organization.id}/${taskId}` }),
    }),
  );
  if (error && error.message !== 'Task not found') throw new Error('Failed to terminate task');

  await prisma.tasks.update({ where: { id: taskId, organizationId: organization.id }, data: { status: 'terminated' } });
});

export const resumeTask = withUserAuth(
  async ({ organization, args }: AuthWrapperContext<{ taskId: string; input: string }>) => {
    const { taskId, input } = args;

    // Ensure the task exists and belongs to the organization.
    const existingTask = await prisma.tasks.findUnique({
      where: { id: taskId, organizationId: organization.id },
      include: {
        progresses: {
          orderBy: { index: 'desc' },
          take: 1,
        },
      }
    });
    if (!existingTask) {
      throw new Error('Task not found or not authorized');
    }

    // Store the user input in the database
    const lastProgress = existingTask.progresses[0];
    const newProgress = await prisma.taskProgresses.create({
      data: {
        taskId: taskId,
        organizationId: organization.id,
        type: 'agent:lifecycle:interaction',
        content: { request: input },
        index: (lastProgress?.index || 0) + 1,
        round: lastProgress?.round || 0,
        step: (lastProgress?.step || 0) + 1,
      },
    });

    // Update task status to processing
    await prisma.tasks.update({
      where: { id: taskId },
      data: { status: 'processing' }
    });

    // The Python API endpoint is /tasks/resume with organization_id and task_id as query parameters.
    const [error, response] = await to(
      fetch(`${MANUS_URL}/tasks/resume?organization_id=${organization.id}&task_id=${taskId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input }), // FastAPI route expects ResumeTaskInput(input: str)
      }).then(async res => {
        if (!res.ok) {
          const errorData = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(`Error resuming task: ${errorData.detail || res.statusText}`);
        }
        return res.json() as Promise<{ message: string; task_id: string }>;
      })
    );

    if (error) {
      // Update task status to failed if there's an error
      await prisma.tasks.update({
        where: { id: taskId },
        data: { status: 'failed' }
      });
      throw error;
    }

    // Handle event stream in background
    handleTaskEvents(taskId, response.task_id, organization.id).catch(error => {
      console.error('Failed to handle task events:', error);
    });

    return { id: taskId, outId: response.task_id };
  }
);

export const shareTask = withUserAuth(async ({ organization, args }: AuthWrapperContext<{ taskId: string; expiresAt: number }>) => {
  const { taskId, expiresAt } = args;
  const task = await prisma.tasks.findUnique({ where: { id: taskId, organizationId: organization.id } });
  if (!task) throw new Error('Task not found');
  await prisma.tasks.update({ where: { id: taskId }, data: { shareExpiresAt: new Date(expiresAt) } });
});

export const getSharedTask = async ({ taskId }: { taskId: string }) => {
  const task = await prisma.tasks.findUnique({
    where: { id: taskId },
    include: { progresses: { orderBy: { index: 'asc' } } },
  });
  if (!task) throw new Error('Task not found');
  if (task.shareExpiresAt && task.shareExpiresAt < new Date()) {
    throw new Error('Task Share Link expired');
  }
  return { data: task, error: null };
};

export const deleteTask = withUserAuth(async ({ organization, args }: AuthWrapperContext<{ taskId: string }>) => {
  const { taskId } = args;

  // Check if task exists and belongs to the organization
  const task = await prisma.tasks.findUnique({
    where: {
      id: taskId,
      organizationId: organization.id
    },
  });

  if (!task) {
    throw new Error('Task not found');
  }

  // Delete task progresses first (due to foreign key constraint)
  await prisma.taskProgresses.deleteMany({
    where: { taskId }
  });

  // Delete the task
  await prisma.tasks.delete({
    where: { id: taskId }
  });

  // Delete the task directory from the workspace
  const workspaceDir = process.env.WORKSPACE_ROOT_PATH;
  if (workspaceDir) {
    const taskDirPath = path.join(workspaceDir, organization.id, taskId);
    try {
      await fsPromises.rm(taskDirPath, { recursive: true, force: true });
      console.log(`Successfully deleted task directory: ${taskDirPath}`);
    } catch (error) {
      console.error(`Failed to delete task directory ${taskDirPath}:`, error);
      // Decide if this error should be propagated or just logged
      // For now, we'll log it and let the operation be considered successful
      // as the primary goal (DB deletion) was achieved.
    }
  } else {
    console.warn('WORKSPACE_DIR environment variable is not set. Cannot delete task directory.');
  }

  return { success: true };
});

// Handle event stream in background
async function handleTaskEvents(taskId: string, outId: string, organizationId: string) {
  const streamResponse = await fetch(`${MANUS_URL}/tasks/${outId}/events`);
  const reader = streamResponse.body?.getReader();
  if (!reader) throw new Error('Failed to get response stream');

  const decoder = new TextDecoder();

  const taskProgresses = await prisma.taskProgresses.findMany({ where: { taskId }, orderBy: { index: 'asc' } });
  const rounds = taskProgresses.map(progress => progress.round);
  const round = Math.max(...rounds, 1);
  let messageIndex = taskProgresses.length || 0;
  let buffer = '';
  try {
    while (true) {
      const { done, value } = await reader.read();

      if (value) {
        buffer += decoder.decode(value, { stream: true });
      }

      const lines = buffer.split('\n');
      // Keep the last line (might be incomplete) if not the final read
      buffer = done ? '' : lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ') || line === 'data: [DONE]') continue;

        try {
          const parsed = JSON.parse(line.slice(6));
          const { event_name, step, content } = parsed;

          // Write message to database
          await prisma.taskProgresses.create({
            data: { taskId, organizationId, index: messageIndex++, step, round, type: event_name, content },
          });

          // If complete message, update task status
          if (event_name === 'agent:lifecycle:complete') {
            await prisma.tasks.update({
              where: { id: taskId },
              data: { status: 'completed' },
            });
            return;
          }
          if (event_name === 'agent:lifecycle:terminating') {
            await prisma.tasks.update({
              where: { id: taskId },
              data: { status: 'terminating' },
            });
          }
          if (event_name === 'agent:lifecycle:terminated') {
            await prisma.tasks.update({
              where: { id: taskId },
              data: { status: 'terminated' },
            });
            return;
          }
        } catch (error) {
          console.error('Failed to process message:', error);
        }
      }

      if (done) break;
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Build full url for MCP SSE
 *
 * url is stored in the config of the tool schema
 * query is stored in the tool
 * so we need to build the full url with query parameters
 *
 * @param url - The base URL
 * @param query - The query parameters
 * @returns The full URL with query parameters
 */
const buildMcpSseFullUrl = (url: string, query: Record<string, string>) => {
  if (!url) return '';
  let fullUrl = url;
  if (Object.keys(query).length > 0) {
    const queryParams = new URLSearchParams(query);
    fullUrl = `${fullUrl}${fullUrl.includes('?') ? '&' : '?'}${queryParams.toString()}`;
  }
  return fullUrl;
};
