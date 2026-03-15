import { API_URL } from '../constants'
import type { Message } from '../types'

interface StreamCallbacks {
  onStatus: (status: string) => void
  onComplete: (messages: Message[], sources: { title: string; url: string }[]) => void
  onError: (error: Error) => void
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

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    const chunk = decoder.decode(value)
    const lines = chunk.split('\n')

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6))

        if (data.status) {
          callbacks.onStatus(data.status)
        }

        if (data.done) {
          callbacks.onComplete(data.messages, data.sources || [])
        }
      }
    }
  }
}
