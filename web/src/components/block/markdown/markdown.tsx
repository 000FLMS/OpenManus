import { cn } from '@/lib/utils';
import React from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';

export const Markdown: React.FC<{ children: unknown; className?: string }> = ({ children, className }) => {
  console.log('children', children);
  const content = typeof children === 'string' ? children : '';
  return (
    <div className={cn('markdown-body mt-2 rounded-md p-4', className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={{
          a: ({ href, children }) => {
            return (
              <a href={href} target="_blank" rel="noopener noreferrer">
                {children}
              </a>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};
