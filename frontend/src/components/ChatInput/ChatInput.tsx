import { useState, useRef, useEffect } from 'react'
import type { KeyboardEvent, ChangeEvent } from 'react'
import { MAX_HEIGHT } from '../../constants'
import { DeepResearchIcon, SendIcon } from '../Icons'
import './ChatInput.css'

interface ChatInputProps {
  onSubmit: (content: string, deepResearch: boolean) => void
  isLoading: boolean
}

export function ChatInput({ onSubmit, isLoading }: ChatInputProps) {
  const [input, setInput] = useState('')
  const [isScrollable, setIsScrollable] = useState(false)
  const [deepResearch, setDeepResearch] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

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
    onSubmit(input.trim(), deepResearch)
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

  return (
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
  )
}

