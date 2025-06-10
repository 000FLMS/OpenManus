import { Markdown } from '@/components/block/markdown/markdown';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Tools } from '@prisma/client';
import { forwardRef, useImperativeHandle, useRef, useState } from 'react';
import { ToolConfigDialog, ToolConfigDialogRef } from './tool-config-dialog';
import { deleteTool } from '@/actions/tools';

export interface ToolInfoDialogRef {
  showInfo: (tool: Tools) => void;
}

interface ToolInfoDialogProps {
  onConfigSuccess: () => void;
}

export const ToolInfoDialog = forwardRef<ToolInfoDialogRef, ToolInfoDialogProps>((props, ref) => {
  const [open, setOpen] = useState(false);
  const [tool, setTool] = useState<Tools>();
  const toolConfigDialogRef = useRef<ToolConfigDialogRef>(null);
  useImperativeHandle(ref, () => ({
    showInfo: (tool: Tools) => {
      setTool(tool);
      setOpen(true);
    },
  }));

  if (!tool) {
    return null;
  }
  console.log('ToolInfoDialog', tool);

  return (
    <>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent style={{ height: '800px', width: '1200px', maxWidth: 1200, overflowY: 'hidden', display: 'flex', flexDirection: 'column' }}>
          <DialogHeader>
            <DialogTitle className="text-2xl font-bold">{tool.name}</DialogTitle>
            <div className="mt-2 flex gap-2">
              <Button variant="default" size="sm" className="w-full sm:w-auto" onClick={() => toolConfigDialogRef.current?.showConfig(tool!)}>
                Install Tool
              </Button>
              <Button variant="destructive" size="sm" className="w-full sm:w-auto" onClick={() => handleDeleteTool(tool.id)}>
                Delete Tool
              </Button>
            </div>
          </DialogHeader>
          <div className="flex-1 overflow-auto">
            <Markdown>{tool.description}</Markdown>
            <div className="mt-4">
              <h3 className="text-lg font-semibold">Args:</h3>
              <pre className="bg-gray-100 p-2 rounded text-sm overflow-auto">
                {JSON.stringify(tool.args, null, 4)}
              </pre>
            </div>
            <div className="mt-4">
              <h3 className="text-lg font-semibold">Env Schema:</h3>
              <pre className="bg-gray-100 p-2 rounded text-sm overflow-auto">
                {JSON.stringify(tool.envSchema, null, 4)}
              </pre>
            </div>
            <div className="mt-4">
              <h3 className="text-lg font-semibold">Command:</h3>
              <pre className="bg-gray-100 p-2 rounded text-sm overflow-auto">
                {tool.command}
              </pre>
            </div>
          </div>
        </DialogContent>
      </Dialog>
      <ToolConfigDialog ref={toolConfigDialogRef} onSuccess={props.onConfigSuccess} />
    </>
  );

  function handleDeleteTool(toolId: string) {
    if (confirm('Are you sure you want to delete this tool?')) {
      deleteTool({ toolId })
        .then(() => {
          setOpen(false);
          props.onConfigSuccess(); // Trigger a refresh or update callback
        })
        .catch(error => {
          console.error('Failed to delete tool:', error);
          alert('Failed to delete tool. Please try again later.');
        });
    }
  }
});
