import { useState, useRef, useEffect } from 'react'
import type { KeyboardEvent, ChangeEvent } from 'react'
import { WelcomeScreen } from './components/WelcomeScreen'
import { MessageList } from './components/MessageList'
import { DeepResearchIcon, SendIcon } from './components/Icons'
import { useChat } from './hooks/useChat'
import { MAX_HEIGHT } from './constants'
import './styles/global.css'
import './components/ChatInput/ChatInput.css'

function App() {
  const [input, setInput] = useState('')
  const [isScrollable, setIsScrollable] = useState(false)
  const [deepResearch, setDeepResearch] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const {
    messages,
    isLoading,
    isStreaming,
    streamingContent,
    status,
    lastMessageSources,
    submitMessage,
  } = useChat()

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

  const handleSubmit = () => {
    if (!input.trim() || isLoading) return
    submitMessage(input.trim(), deepResearch)
    setInput('')
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

  const handleStarterQuestion = (question: string, isDeepResearch: boolean) => {
    if (isDeepResearch) {
      setDeepResearch(true)
    }
    submitMessage(question, isDeepResearch)
  }

  return (
    <div className={`app ${hasStarted ? 'started' : ''}`}>
      {!hasStarted && (
        <WelcomeScreen onSelectQuestion={handleStarterQuestion} />
      )}

      {hasStarted && (
        <MessageList
          messages={messages}
          isLoading={isLoading}
          isStreaming={isStreaming}
          streamingContent={streamingContent}
          status={status}
          lastMessageSources={lastMessageSources}
        />
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
                <DeepResearchIcon />
              </button>
              <span className="tooltip">Deep research</span>
            </div>
            <button
              className="submit-button"
              onClick={handleSubmit}
              disabled={!input.trim() || isLoading}
            >
              <SendIcon />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
