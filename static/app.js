class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chat-icon-btn-side'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button'),
            messageInput: document.querySelector('.message-input'),
            minimizeButton: document.querySelector('.minimize-btn'),
            suggestionButtons: document.querySelectorAll('.suggestion-btn'),
            notificationBadge: document.querySelector('.notification-badge-side')
        }

        this.state = false;
        this.messages = [];
        
        // Add initial greeting (as shown in your design)
        this.addMessage("Sam", "Hi. My name is Sam. How can I help you?");
    }

    display() {
        const {openButton, chatBox, sendButton, messageInput, minimizeButton, 
               suggestionButtons, notificationBadge} = this.args;

        // Update the initial chat display
        this.updateChatText(chatBox);

        // Open/close chatbox from side icon
        openButton.addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggleState(chatBox);
            if (this.state) {
                notificationBadge.style.display = 'none';
            }
        });

        // Send message
        sendButton.addEventListener('click', () => this.onSendButton(chatBox));

        // Enter key
        messageInput.addEventListener("keyup", ({key}) => {
            if (key === "Enter") {
                this.onSendButton(chatBox);
            }
        });

        // Close chatbox
        minimizeButton.addEventListener('click', () => {
            this.toggleState(chatBox);
        });

        // Quick reply buttons
        suggestionButtons.forEach(button => {
            button.addEventListener('click', () => {
                const text = button.textContent;
                messageInput.value = text;
                this.onSendButton(chatBox);
            });
        });

        // Close chatbox when clicking outside (only if open)
        document.addEventListener('click', (e) => {
            if (this.state && 
                !chatBox.contains(e.target) && 
                !openButton.contains(e.target)) {
                this.toggleState(chatBox);
            }
        });

        // Auto-open on first visit (optional)
        setTimeout(() => {
            if (!this.state && !localStorage.getItem('chatVisited')) {
                this.toggleState(chatBox);
                localStorage.setItem('chatVisited', 'true');
            }
        }, 2000);
    }

    toggleState(chatbox) {
        this.state = !this.state;
        
        if (this.state) {
            chatbox.classList.add('chatbox--active');
            this.args.messageInput.focus();
            this.showTypingIndicator(chatbox, false);
        } else {
            chatbox.classList.remove('chatbox--active');
        }
    }

    onSendButton(chatbox) {
        const textField = this.args.messageInput;
        let text1 = textField.value.trim();
        
        if (text1 === "") {
            return;
        }

        // Add user message
        this.addMessage("User", text1);
        
        // Clear input
        textField.value = '';
        
        // Update chat immediately
        this.updateChatText(chatbox);
        
        // Show typing indicator
        this.showTypingIndicator(chatbox, true);
        
        // Send to backend (KEEPING YOUR ORIGINAL API CALL)
        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        })
        .then(r => r.json())
        .then(r => {
            this.showTypingIndicator(chatbox, false);
            
            // Add bot response
            this.addMessage("Sam", r.answer);
            this.updateChatText(chatbox);
            
            // Auto-scroll to bottom
            this.scrollToBottom(chatbox);
            
            // Show notification if chat is minimized
            if (!this.state) {
                this.args.notificationBadge.style.display = 'flex';
                this.args.notificationBadge.textContent = '!';
            }
        })
        .catch((error) => {
            console.error('Error:', error);
            this.showTypingIndicator(chatbox, false);
            
            // Show error message
            this.addMessage("Sam", "I apologize, but I'm having trouble connecting right now. Please try again in a moment.");
            this.updateChatText(chatbox);
        });
    }

    addMessage(name, message) {
        this.messages.push({ name, message });
    }

    showTypingIndicator(chatbox, show) {
        const messagesDiv = chatbox.querySelector('.chatbox__messages');
        const typingDiv = messagesDiv.querySelector('.typing-indicator');
        
        if (show && !typingDiv) {
            const typingHTML = `
                <div class="messages__item messages__item--visitor">
                    <div class="typing-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            `;
            messagesDiv.insertAdjacentHTML('afterbegin', typingHTML);
            this.scrollToBottom(chatbox);
        } else if (!show && typingDiv) {
            typingDiv.remove();
        }
    }

    updateChatText(chatbox) {
        const messagesDiv = chatbox.querySelector('.chatbox__messages');
        let html = '';
        
        // Show messages in reverse order (newest at bottom)
        [...this.messages].reverse().forEach((item, index) => {
            if (item.name === "Sam") {
                html += `
                    <div class="messages__item messages__item--visitor">
                        <div class="message-content">
                            ${item.message}
                        </div>
                    </div>
                `;
            } else {
                html += `
                    <div class="messages__item messages__item--operator">
                        <div class="message-content">
                            ${item.message}
                        </div>
                    </div>
                `;
            }
        });

        messagesDiv.innerHTML = html;
        this.scrollToBottom(chatbox);
    }

    scrollToBottom(chatbox) {
        const messagesDiv = chatbox.querySelector('.chatbox__messages');
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }
}

// Initialize chatbox
const chatbox = new Chatbox();
chatbox.display();

// Add CSS for typing animation
const style = document.createElement('style');
style.textContent = `
    .typing-dots {
        display: flex;
        gap: 5px;
        padding: 10px;
    }
    
    .typing-dots span {
        width: 8px;
        height: 8px;
        background: #888;
        border-radius: 50%;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
    .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes typing {
        0%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-5px); }
    }
`;
document.head.appendChild(style);