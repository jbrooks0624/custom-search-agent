import { useState, useRef, useEffect, KeyboardEvent, ChangeEvent } from 'react'
import ReactMarkdown from 'react-markdown'
import './App.css'

const MAX_HEIGHT = 200
const API_URL = 'http://localhost:8000'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

interface Source {
  title: string
  url: string
}

interface ChatResponse {
  messages: Message[]
  iterations: number
  total_ms: number
  sources: Source[]
}

function App() {
  const [input, setInput] = useState('')
  const [isScrollable, setIsScrollable] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [deepResearch, setDeepResearch] = useState(false)
  const [status, setStatus] = useState('')
  const [sources, setSources] = useState<Source[]>([])
  const [lastMessageSources, setLastMessageSources] = useState<Map<number, Source[]>>(new Map())
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const hasStarted = messages.length > 0

  const adjustHeight = () => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      const scrollHeight = textarea.scrollHeight
      const shouldScroll = scrollHeight > MAX_HEIGHT
      setIsScrollable(shouldScroll)
      textarea.style.height = `${Math.min(scrollHeight, MAX_HEIGHT)}px`
    }
  }

  useEffect(() => {
    adjustHeight()
  }, [input])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = async () => {
    if (!input.trim() || isLoading) return
    
    const userMessage: Message = { role: 'user', content: input.trim() }
    const updatedMessages = [...messages, userMessage]
    setMessages(updatedMessages)
    setInput('')
    setIsLoading(true)
    setStatus('')

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: updatedMessages,
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
              setStatus(data.status)
            }
            
            if (data.done) {
              setMessages(data.messages)
              if (data.sources && data.sources.length > 0) {
                const assistantIndex = data.messages.length - 1
                setLastMessageSources(prev => new Map(prev).set(assistantIndex, data.sources))
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Error:', error)
      const errorMessage: Message = {
        role: 'assistant',
        content: 'Sorry, there was an error processing your request. Please try again.',
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
      setStatus('')
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
  }

  return (
    <div className={`app ${hasStarted ? 'started' : ''}`}>
      {!hasStarted && (
        <div className="welcome">
          <h1>Custom Search Agent</h1>
          <p className="subtitle">Enable deep research mode for comprehensive, well-researched answers</p>
        </div>
      )}

      {hasStarted && (
        <div className="messages-container">
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.role}`}>
              {message.role === 'user' ? (
                <div 
                  className={`message-bubble ${message.content.split('\n').length > 10 || message.content.length > 500 ? 'scrollable' : ''}`}
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
          <div ref={messagesEndRef} />
        </div>
      )}

      <div className="chat-input-container">
        <div className="chat-input-wrapper">
          <textarea
            ref={textareaRef}
            className={`chat-input ${isScrollable ? 'scrollable' : ''}`}
            value={input}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything..."
            rows={1}
          />
          <div className="input-buttons">
            <div className="deep-research-wrapper">
              <button
                className={`deep-research-button ${deepResearch ? 'active' : ''}`}
                onClick={() => setDeepResearch(!deepResearch)}
              >
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <circle cx="11" cy="11" r="8" />
                  <path d="M21 21l-4.35-4.35" />
                  <path d="M11 8v6M8 11h6" />
                </svg>
              </button>
              <span className="tooltip">Deep research</span>
            </div>
            <button 
              className="submit-button"
              onClick={handleSubmit}
              disabled={!input.trim() || isLoading}
            >
              <svg 
                width="20" 
                height="20" 
                viewBox="0 0 24 24" 
                fill="none" 
                stroke="currentColor" 
                strokeWidth="2" 
                strokeLinecap="round" 
                strokeLinejoin="round"
              >
                <path d="M12 19V5M5 12l7-7 7 7" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
