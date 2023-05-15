(function() {
  document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chat-form");
    const chatMessages = document.getElementById("chat-messages");
    const userInput = document.getElementById("user-input");
    const loadingOverlay = document.getElementById("loading-overlay");
    const sessionId = Math.random().toString(36).substring(7);
    const socket = io({ extraHeaders: { sessionId } });

    chatForm.addEventListener("submit", handleSubmitForm);

    userInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        insertNewLine();
      } else if (event.key === "Enter" && event.shiftKey) {
        // Shift + Enter: Submit the form
        event.preventDefault();
        handleSubmitForm(event);
      }
    });

    function handleSubmitForm(event) {
      event.preventDefault();
      const userMessage = userInput.value;
      if (userMessage.trim()) {
        const userMessageElement = document.createElement("div");
        userMessageElement.classList.add("message", "user-message");
        const preElement = document.createElement("pre");
        preElement.textContent = userMessage;
        userMessageElement.appendChild(preElement);
        chatMessages.appendChild(userMessageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        loadingOverlay.classList.add("show");
        socket.emit("user_message", { session_id: sessionId, message: userMessage });
        userInput.value = "";
      }
    }

    function insertNewLine() {
      const { selectionStart, selectionEnd, value } = userInput;
      const newValue =
        value.slice(0, selectionStart) + "\n" + value.slice(selectionEnd);

      userInput.value = newValue;
      userInput.selectionStart = userInput.selectionEnd =
        selectionStart + 1;
    }

    socket.on("bot_response", handleBotResponse);

    function handleBotResponse(data) {
      console.log("Received bot_response event:", data);
      if (data.sessionId === sessionId) {
        console.log(`Bot response received: ${data.response}`);
        const botMessageElement = document.createElement("div");
        botMessageElement.classList.add("message", "bot-message");
        botMessageElement.textContent = `${data.response}`;
        chatMessages.appendChild(botMessageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        loadingOverlay.classList.remove("show");
      } else {
        console.log(
          "Received bot_response for different sessionId:",
          data.sessionId
        );
      }
    }
  });
})();
