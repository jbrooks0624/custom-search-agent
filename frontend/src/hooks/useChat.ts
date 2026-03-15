import { useState, useCallback } from 'react'
import type { Message, Source } from '../types'
import { sendChatMessage, type ChatError } from '../api/chat'
import { MAX_MESSAGES, MAX_MESSAGE_LENGTH } from '../constants'

const ERROR_MESSAGES: Record<string, string> = {
  rate_limit: 'Too many requests. Please wait a moment and try again.',
  connection_error: 'Unable to connect to the server. Please check your connection.',
  api_error: 'The AI service encountered an error. Please try again.',
  internal_error: 'Something went wrong. Please try again.',
  message_too_long: `Message exceeds maximum length of ${MAX_MESSAGE_LENGTH.toLocaleString()} characters.`,
  conversation_limit: `Conversation limit reached (${MAX_MESSAGES} messages). Please start a new conversation.`,
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
    // Validate message length
    if (content.length > MAX_MESSAGE_LENGTH) {
      handleError({ code: 'message_too_long', message: ERROR_MESSAGES.message_too_long })
      return
    }

    // Check conversation limit (need room for user message + assistant response)
    if (messages.length >= MAX_MESSAGES - 1) {
      handleError({ code: 'conversation_limit', message: ERROR_MESSAGES.conversation_limit })
      return
    }

    const userMessage: Message = { role: 'user', content }
    const updatedMessages = [...messages, userMessage]
    setMessages(updatedMessages)
    setIsLoading(true)
    setIsStreaming(false)
    setStreamingContent('')
    setStatus('')

    try {
      await sendChatMessage(updatedMessages, deepResearch, {
        onStatus: (newStatus) => setStatus(newStatus),
        onContentDelta: (delta) => {
          setIsStreaming(true)
          setStreamingContent(prev => prev + delta)
        },
        onComplete: (finalMessages, sources) => {
          setIsLoading(false)
          setIsStreaming(false)
          setStatus('')
          setStreamingContent('')
          setMessages(finalMessages)
          if (sources.length > 0) {
            const assistantIndex = finalMessages.length - 1
            setLastMessageSources(prev => new Map(prev).set(assistantIndex, sources))
          }
        },
        onError: (error) => {
          handleError(error)
        },
      })
    } catch (error) {
      console.error('Error:', error)
      handleError(error as Error)
    }
  }, [messages, handleError])

  const resetConversation = useCallback(() => {
    setMessages([])
    setIsLoading(false)
    setIsStreaming(false)
    setStreamingContent('')
    setStatus('')
    setLastMessageSources(new Map())
  }, [])

  const isAtMessageLimit = messages.length >= MAX_MESSAGES - 1

  return {
    messages,
    isLoading,
    isStreaming,
    streamingContent,
    status,
    lastMessageSources,
    submitMessage,
    resetConversation,
    isAtMessageLimit,
  }
}
