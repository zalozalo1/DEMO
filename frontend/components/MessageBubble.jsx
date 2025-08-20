export default function MessageBubble({ role, children }) {
  const isUser = role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-2 shadow-sm mb-3 whitespace-pre-wrap ${
          isUser ? "bg-blue-600 text-white" : "bg-white text-gray-900 border border-gray-200"
        }`}
      >
        {children}
      </div>
    </div>
  );
}