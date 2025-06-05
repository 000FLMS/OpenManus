'use client';

import { getTask, restartTask, terminateTask, resumeTask } from '@/actions/tasks';
import { ChatInput } from '@/components/features/chat/input';
import { ChatMessages } from '@/components/features/chat/messages';
import { ChatPreview } from '@/components/features/chat/preview';
import { usePreviewData } from '@/components/features/chat/preview/store';
import { aggregateMessages } from '@/lib/chat-messages';
import { Message } from '@/lib/chat-messages/types';
import { useParams, useRouter } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';
import { toast } from 'sonner';

export default function ChatPage() {
  const params = useParams();
  const router = useRouter();
  const taskId = params.taskid as string;

  const { setData: setPreviewData } = usePreviewData();

  const [isNearBottom, setIsNearBottom] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isThinking, setIsThinking] = useState(false);
  const [isTerminating, setIsTerminating] = useState(false);
  const [isPausedForInput, setIsPausedForInput] = useState(false);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const shouldAutoScroll = isNearBottom;

  const handleScroll = () => {
    if (messagesContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current;
      const isNearBottom = Math.abs(scrollHeight - scrollTop - clientHeight) < 10;
      setIsNearBottom(isNearBottom);
    }
  };

  const scrollToBottom = () => {
    if (messagesContainerRef.current && shouldAutoScroll) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
    }
  };

  const refreshTask = async () => {
    const res = await getTask({ taskId });
    if (res.error || !res.data) {
      console.error('Error fetching task:', res.error);
      toast.error('Task not found');
      router.push('/');
      return;
    }
    setMessages(res.data.progresses.map(step => ({ ...step, index: step.index! || 0, type: step.type as any, role: 'assistant' as const })));
    if (shouldAutoScroll) {
      requestAnimationFrame(scrollToBottom);
      const nextMessage = messages[messages.length - 1];
      if (shouldAutoScroll) {
        if (nextMessage?.type === 'agent:lifecycle:step:think:browser:browse:complete') {
          setPreviewData({
            type: 'browser',
            url: nextMessage.content.url,
            title: nextMessage.content.title,
            screenshot: nextMessage.content.screenshot,
          });
        }
        if (nextMessage?.type === 'agent:lifecycle:step:act:tool:execute:start') {
          setPreviewData({ type: 'tool', toolId: nextMessage.content.id });
        }
      }
    }
    setIsThinking(res.data!.status !== 'completed' && res.data!.status !== 'failed' && res.data!.status !== 'terminated');
    setIsTerminating(res.data!.status === 'terminating');
  };

  useEffect(() => {
    setPreviewData(null);
    if (!taskId) return;
    if (!isPausedForInput) {
      refreshTask();
    }
  }, [taskId]);

  useEffect(() => {
    if (!taskId || !isThinking || isPausedForInput) {
      return;
    }
    const interval = setInterval(refreshTask, 2000);
    return () => {
      clearInterval(interval);
    };
  }, [taskId, isThinking, isPausedForInput, shouldAutoScroll]);

  useEffect(() => {
    if (shouldAutoScroll) {
      requestAnimationFrame(scrollToBottom);
    }
  }, [messages, shouldAutoScroll]);

  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  useEffect(() => {
    if (messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      if (lastMessage.type === 'agent_paused_for_input') {
        setIsPausedForInput(true);
        setIsThinking(false);
        console.log('Agent is paused for input');
      } else if (isPausedForInput && lastMessage.role === 'user') {
        setIsPausedForInput(false);
      }
    }
  }, [messages, isPausedForInput]);

  const handleResumeWithInput = async (userInput: string) => {
    if (!taskId || !isPausedForInput) return;



    try {
      const response = await resumeTask({ taskId, input: userInput });
      if (response.error) {
        throw new Error(response.error);
      }
      // Refresh the task to get the latest messages
      await refreshTask();
      setIsPausedForInput(false);
      setIsThinking(true);
      router.refresh();

    } catch (error) {
      toast.error(`Failed to send input: ${error instanceof Error ? error.message : String(error)}`);
      setIsPausedForInput(true);
      setIsThinking(false);
    }
  };

  const handleSubmit = async (value: { modelId: string; prompt: string; tools: string[]; files: File[]; shouldPlan: boolean }) => {
    try {
      const res = await restartTask({
        taskId,
        modelId: value.modelId,
        prompt: value.prompt,
        tools: value.tools,
        files: value.files,
        shouldPlan: value.shouldPlan,
      });
      if (res.error) {
        console.error('Error restarting task:', res.error);
      }
      setIsThinking(true);
      router.refresh();
    } catch (error) {
      console.error('Error submitting task:', error);
    }
  };

  const handleTerminateTask = async () => {
    await terminateTask({ taskId });
    router.refresh();
  };

  return (
    <div className="flex h-screen w-full flex-row justify-between">
      <div className="flex-1">
        <div className="relative flex h-screen flex-col">
          <div
            ref={messagesContainerRef}
            className="flex-3/5 space-y-4 overflow-y-auto p-4 pb-60"
            style={{
              scrollBehavior: 'smooth',
              overscrollBehavior: 'contain',
            }}
            onScroll={handleScroll}
          >
            <ChatMessages messages={aggregateMessages(messages)} />
          </div>
          <ChatInput
            taskId={taskId}
            status={isTerminating ? 'terminating' : isThinking ? 'thinking' : isPausedForInput ? 'paused' : 'idle'}
            onSubmit={isPausedForInput ? async ({ prompt }) => handleResumeWithInput(prompt) : handleSubmit}
            onTerminate={handleTerminateTask}
          />
        </div>
      </div>
      <div className="min-w-[400px] flex-1 items-center justify-center p-2">
        <ChatPreview taskId={taskId} messages={messages} />
      </div>
    </div>
  );
}
