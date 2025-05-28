import { registerTool } from '@/actions/tools';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Textarea } from '@/components/ui/textarea';
import { zodResolver } from '@hookform/resolvers/zod';
import { forwardRef, useImperativeHandle, useState } from 'react';
import { useForm } from 'react-hook-form';
import { toast } from 'sonner';
import * as z from 'zod';

interface ToolRegisterDialogProps {
  onSuccess?: () => void;
}

export interface ToolRegisterDialogRef {
  showRegister: () => void;
}

const commandTypes = ['docker', 'npx', 'uvx'] as const;

interface BasicInfoFormProps {
  form: ReturnType<typeof useForm<z.infer<typeof basicInfoSchema>>>;
  onSubmit: (values: z.infer<typeof basicInfoSchema>) => void;
}

const BasicInfoForm = ({ form, onSubmit }: BasicInfoFormProps) => {
  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
        <FormField
          control={form.control}
          name="name"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Name</FormLabel>
              <FormControl>
                <Input placeholder="Enter tool name" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="description"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Description</FormLabel>
              <FormControl>
                <Textarea placeholder="Enter tool description" className="min-h-[200px]" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <Button type="submit" className="w-full">
          Next
        </Button>
      </form>
    </Form>
  );
};

interface TechnicalConfigFormProps {
  form: ReturnType<typeof useForm<z.infer<typeof technicalConfigSchema>>>;
  onSubmit: (values: z.infer<typeof technicalConfigSchema>) => void;
}

const TechnicalConfigForm = ({ form, onSubmit }: TechnicalConfigFormProps) => {
  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
        <FormField
          control={form.control}
          name="command"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Command Type</FormLabel>
              <Select onValueChange={field.onChange} defaultValue={field.value}>
                <FormControl>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select a command type" />
                  </SelectTrigger>
                </FormControl>
                <SelectContent>
                  {commandTypes.map(type => (
                    <SelectItem key={type} value={type}>
                      {type}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="args"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Args (JSON)</FormLabel>
              <FormControl>
                <Textarea placeholder="Enter tool args in JSON format" className="min-h-[150px] font-mono" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="envSchema"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Environment Schema (JSON)</FormLabel>
              <FormControl>
                <Textarea placeholder="Enter environment schema in JSON format" className="min-h-[150px] font-mono" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <Button type="submit" className="w-full">
          Register
        </Button>
      </form>
    </Form>
  );
};

const basicInfoSchema = z.object({
  name: z.string().min(1, 'Name is required'),
  description: z.string().min(1, 'Description is required'),
});

const technicalConfigSchema = z.object({
  command: z.enum(commandTypes, {
    required_error: 'Please select a command type',
  }),
  args: z.string().refine(
    val => {
      try {
        JSON.parse(val);
        return true;
      } catch {
        return false;
      }
    },
    {
      message: 'Invalid JSON format for args',
    },
  ),
  envSchema: z.string().refine(
    val => {
      try {
        JSON.parse(val);
        return true;
      } catch {
        return false;
      }
    },
    {
      message: 'Invalid JSON format for envSchema',
    },
  ),
});

export const ToolRegisterDialog = forwardRef<ToolRegisterDialogRef, ToolRegisterDialogProps>(({ onSuccess }, ref) => {
  const [open, setOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('basic');

  const basicInfoForm = useForm<z.infer<typeof basicInfoSchema>>({
    resolver: zodResolver(basicInfoSchema),
    defaultValues: {
      name: '',
      description: '',
    },
  });

  const technicalConfigForm = useForm<z.infer<typeof technicalConfigSchema>>({
    resolver: zodResolver(technicalConfigSchema),
    defaultValues: {
      command: 'docker',
      args: '',
      envSchema: '',
    },
  });

  useImperativeHandle(ref, () => ({
    showRegister: () => {
      setOpen(true);
    },
  }));

  const handleBasicInfoSubmit = (values: z.infer<typeof basicInfoSchema>) => {
    setActiveTab('technical');
  };

  const handleTechnicalConfigSubmit = async (values: z.infer<typeof technicalConfigSchema>) => {
    try {
      const toolData = {
        ...basicInfoForm.getValues(),
        command: values.command,
        args: JSON.parse(values.args),
        envSchema: JSON.parse(values.envSchema),
      };

      await registerTool(toolData);

      toast.success('Success', {
        description: 'Tool registered successfully',
      });
      onSuccess?.();
    } catch (error) {
      toast.error('Failed', {
        description: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  };

  const handleOpenChange = (open: boolean) => {
    setOpen(open);
    if (!open) {
      basicInfoForm.reset();
      technicalConfigForm.reset();
      setActiveTab('basic');
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Register New Tool</DialogTitle>
        </DialogHeader>
        <div className="mt-4 flex-1">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="basic">Basic Information</TabsTrigger>
              <TabsTrigger value="technical" disabled={!basicInfoForm.formState.isValid}>
                Configuration
              </TabsTrigger>
            </TabsList>
            <TabsContent value="basic">
              <BasicInfoForm form={basicInfoForm} onSubmit={handleBasicInfoSubmit} />
            </TabsContent>
            <TabsContent value="technical">
              <TechnicalConfigForm form={technicalConfigForm} onSubmit={handleTechnicalConfigSubmit} />
            </TabsContent>
          </Tabs>
        </div>
      </DialogContent>
    </Dialog>
  );
});
