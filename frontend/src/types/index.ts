export interface Message {
  role: 'user' | 'assistant'
  content: string
}

export interface Source {
  title: string
  url: string
}

export interface ChatResponse {
  messages: Message[]
  iterations: number
  total_ms: number
  sources: Source[]
}
