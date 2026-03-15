import { useState, useCallback } from 'react'
import type { Message, Source } from '../types'
import { sendChatMessage, type ChatError } from '../api/chat'

const ERROR_MESSAGES: Record<string, string> = {
  rate_limit: 'Too many requests. Please wait a moment and try again.',
  connection_error: 'Unable to connect to the server. Please check your connection.',
  api_error: 'The AI service encountered an error. Please try again.',
  internal_error: 'Something went wrong. Please try again.',
}

function getErrorMessage(error: ChatError): string {
  const base = ERROR_MESSAGES[error.code] || error.message
  if (error.retryAfter) {
    return `${base} (retry in ${error.retryAfter}s)`
  }
  return base
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingContent, setStreamingContent] = useState('')
  const [status, setStatus] = useState('')
  const [lastMessageSources, setLastMessageSources] = useState<Map<number, Source[]>>(new Map())

  const fakeStreamResponse = useCallback((
    fullContent: string,
    finalMessages: Message[],
    sources: Source[]
  ) => {
    setIsStreaming(true)
    setStreamingContent('')

    let charIndex = 0
    const streamInterval = setInterval(() => {
      const charsPerTick = Math.floor(Math.random() * 3) + 2
      charIndex += charsPerTick

      if (charIndex >= fullContent.length) {
        clearInterval(streamInterval)
        setIsStreaming(false)
        setStreamingContent('')
        setMessages(finalMessages)
        if (sources.length > 0) {
          const assistantIndex = finalMessages.length - 1
          setLastMessageSources(prev => new Map(prev).set(assistantIndex, sources))
        }
      } else {
        setStreamingContent(fullContent.slice(0, charIndex))
      }
    }, 10)
  }, [])

  const handleError = useCallback((error: ChatError | Error) => {
    const message = 'code' in error
      ? getErrorMessage(error as ChatError)
      : 'Sorry, there was an error processing your request. Please try again.'
    
    const errorMessage: Message = {
      role: 'assistant',
      content: message,
    }
    setMessages(prev => [...prev, errorMessage])
    setIsLoading(false)
    setStatus('')
  }, [])

  const submitMessage = useCallback(async (
    content: string,
    deepResearch: boolean
  ) => {
    const userMessage: Message = { role: 'user', content }
    const updatedMessages = [...messages, userMessage]
    setMessages(updatedMessages)
    setIsLoading(true)
    setStatus('')

    try {
      await sendChatMessage(updatedMessages, deepResearch, {
        onStatus: (newStatus) => setStatus(newStatus),
        onComplete: (finalMessages, sources) => {
          setIsLoading(false)
          setStatus('')
          const fullContent = finalMessages[finalMessages.length - 1].content
          fakeStreamResponse(fullContent, finalMessages, sources)
        },
        onError: (error) => {
          handleError(error)
        },
      })
    } catch (error) {
      console.error('Error:', error)
      handleError(error as Error)
    }
  }, [messages, fakeStreamResponse, handleError])

  return {
    messages,
    isLoading,
    isStreaming,
    streamingContent,
    status,
    lastMessageSources,
    submitMessage,
  }
}
