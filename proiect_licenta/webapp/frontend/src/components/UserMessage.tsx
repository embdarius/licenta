// The patient's typed answer, right-aligned.
export default function UserMessage({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[80%] rounded-lg rounded-tr-none bg-clinical px-3 py-2 text-sm text-white">
        {text}
      </div>
    </div>
  );
}
