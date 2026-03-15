import { API_URL } from '../constants'
import type { Message } from '../types'

export interface ChatError {
  code: string
  message: string
  retryAfter?: number
}

interface StreamCallbacks {
  onStatus: (status: string) => void
  onComplete: (messages: Message[], sources: { title: string; url: string }[]) => void
  onError: (error: ChatError) => void
}

function processSSELine(
  line: string,
  callbacks: StreamCallbacks
): void {
  if (!line.startsWith('data: ')) return

  try {
    const data = JSON.parse(line.slice(6))

    if (data.error) {
      callbacks.onError({
        code: data.code,
        message: data.message,
        retryAfter: data.retry_after,
      })
      return
    }

    if (data.status) {
      callbacks.onStatus(data.status)
    }

    if (data.done) {
      callbacks.onComplete(data.messages, data.sources || [])
    }
  } catch (e) {
    console.error('Failed to parse SSE data:', e, 'Line:', line)
  }
}

export async function sendChatMessage(
  messages: Message[],
  deepResearch: boolean,
  callbacks: StreamCallbacks
): Promise<void> {
  const response = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      messages,
      deep_research: deepResearch,
    }),
  })

  if (!response.ok) {
    throw new Error('Failed to get response')
  }

  const reader = response.body?.getReader()
  const decoder = new TextDecoder()

  if (!reader) {
    throw new Error('No response body')
  }

  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    
    if (done) {
      if (buffer.trim()) {
        processSSELine(buffer.trim(), callbacks)
      }
      break
    }

    buffer += decoder.decode(value, { stream: true })

    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      const trimmed = line.trim()
      if (trimmed) {
        processSSELine(trimmed, callbacks)
      }
    }
  }
}
