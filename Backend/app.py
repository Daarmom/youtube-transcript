from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Load the pre-trained model and tokenizer for summarization with legacy=False
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("t5-small")


# Function to get and parse the YouTube video transcript
def get_transcript(video_id):
    try:
        # Fetch the transcript from the YouTubeTranscriptApi
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Parse the transcript data into a single string
        full_transcript = " ".join([entry['text'] for entry in transcript])

        return full_transcript
    except Exception as e:
        return str(e)


# Function to summarize the transcript using T5
def summarize_transcript(transcript_text):
    # Add the T5 specific prefix "summarize: " to the input
    input_text = "summarize: " + transcript_text

    # Tokenize the input text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the summary (with max length and beam search for more coherent summaries)
    summary_ids = model.generate(inputs, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


# Route for fetching the YouTube transcript
@app.route('/transcript', methods=['GET'])
def fetch_transcript():
    # Get the video_id from the request arguments
    video_id = request.args.get('video_id')
    
    if not video_id:
        return jsonify({"error": "You must provide a video_id as a parameter."}), 400

    # Fetch and parse the transcript
    full_transcript = get_transcript(video_id)

    # Return the parsed transcript as a response
    return jsonify({"transcript": full_transcript})


# Route for summarizing the transcript
@app.route('/summarize', methods=['GET'])
def summarize():
    # Get the video_id from the request arguments
    video_id = request.args.get('video_id')

    if not video_id:
        return jsonify({"error": "You must provide a video_id as a parameter."}), 400

    # Fetch the full transcript
    full_transcript = get_transcript(video_id)

    # Summarize the transcript
    summarized_transcript = summarize_transcript(full_transcript)

    # Return the summarized transcript as a response
    return jsonify({"summary": summarized_transcript})


if __name__ == '__main__':
    app.run(debug=True)
