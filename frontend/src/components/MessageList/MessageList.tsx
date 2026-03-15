import { useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import type { Message, Source } from '../../types'
import './MessageList.css'

interface MessageListProps {
  messages: Message[]
  isLoading: boolean
  isStreaming: boolean
  streamingContent: string
  status: string
  lastMessageSources: Map<number, Source[]>
}

export function MessageList({
  messages,
  isLoading,
  isStreaming,
  streamingContent,
  status,
  lastMessageSources,
}: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingContent])

  return (
    <div className="messages-scroll-wrapper">
      <div className="messages-container">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.role}`}>
            {message.role === 'user' ? (
              <div
                className={`message-bubble ${
                  message.content.split('\n').length > 10 || message.content.length > 500
                    ? 'scrollable'
                    : ''
                }`}
              >
                {message.content}
              </div>
            ) : (
              <div className="assistant-content">
                <ReactMarkdown>{message.content}</ReactMarkdown>
                {lastMessageSources.get(index) && (
                  <div className="sources">
                    <span className="sources-label">Sources:</span>
                    {lastMessageSources.get(index)!.map((source, i) => (
                      <a
                        key={i}
                        href={source.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="source-link"
                      >
                        {source.title}
                      </a>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="message assistant">
            <div className="loading-indicator">
              <div className="loading-dots">
                <span className="dot"></span>
                <span className="dot"></span>
                <span className="dot"></span>
              </div>
              {status && <span className="loading-status">{status}</span>}
            </div>
          </div>
        )}

        {isStreaming && (
          <div className="message assistant">
            <div className="assistant-content">
              <ReactMarkdown>{streamingContent}</ReactMarkdown>
              <span className="cursor">|</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>
    </div>
  )
}
