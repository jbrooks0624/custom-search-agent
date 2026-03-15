import { useState, useCallback } from 'react'
import type { Message, Source } from '../types'
import { sendChatMessage } from '../api/chat'

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
          throw error
        },
      })
    } catch (error) {
      console.error('Error:', error)
      const errorMessage: Message = {
        role: 'assistant',
        content: 'Sorry, there was an error processing your request. Please try again.',
      }
      setMessages(prev => [...prev, errorMessage])
      setIsLoading(false)
      setStatus('')
    }
  }, [messages, fakeStreamResponse])

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
