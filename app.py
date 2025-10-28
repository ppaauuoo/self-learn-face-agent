import gradio as gr
import cv2
import numpy as np
import os
import tempfile
from PIL import Image
import io
import base64

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from learning import add_new_embeddings, load_embeddings, search_similar_faces
from deepface import DeepFace

# Initialize LangChain components
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
except Exception as e:
    print(f"Error initializing Gemini model: {e}")
    llm = None

# Global variables for face database
face_index, face_ids = load_embeddings()
pending_face = None
pending_embedding = None


def detect_and_recognize_face(image):
    """Detect and recognize faces in the uploaded image using DeepFace"""
    global face_index, face_ids

    if image is None:
        return None, "No image uploaded", ""

    try:
        # Convert PIL image to numpy array
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        # Convert RGB to BGR for OpenCV processing
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array

        # Save temporary image for DeepFace processing
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            if isinstance(image, Image.Image):
                image.save(tmp.name)
            else:
                cv2.imwrite(tmp.name, img_bgr)
            tmp_path = tmp.name

        try:
            # Use DeepFace to detect faces
            faces = DeepFace.extract_faces(
                img_path=tmp_path, detector_backend="opencv", enforce_detection=False
            )

            if not faces or len(faces) == 0:
                os.unlink(tmp_path)
                return image, "No faces detected in the image", ""

            # Get face bounding boxes and embeddings
            embedding_objs = DeepFace.represent(
                img_path=tmp_path, detector_backend="opencv", enforce_detection=False
            )

            if not embedding_objs:
                os.unlink(tmp_path)
                return image, "Faces detected but couldn't generate embeddings", ""

            # Process first detected face
            face_obj = faces[0]
            embedding = np.array(embedding_objs[0]["embedding"], dtype=np.float32)

            # Get bounding box coordinates (DeepFace returns normalized coordinates)
            x, y, w, h = (
                face_obj["facial_area"]["x"],
                face_obj["facial_area"]["y"],
                face_obj["facial_area"]["w"],
                face_obj["facial_area"]["h"],
            )

            # Draw bounding box
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract face region
            face_region = img_bgr[y : y + h, x : x + w]
            face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)

            # Search for similar faces if database exists
            recognition_result = "Unknown face detected"
            confidence = ""

            if face_index is not None:
                distances, indices = search_similar_faces(embedding, face_index, k=1)

                if (
                    distances is not None
                    and indices is not None
                    and len(distances[0]) > 0
                ):
                    distance = distances[0][0]
                    idx = indices[0][0]

                    # Set a threshold for recognition (lower distance = more similar)
                    threshold = 0.4  # Adjust this based on your needs

                    if distance < threshold:
                        recognized_name = face_ids[idx]
                        recognition_result = f"Recognized: {recognized_name}"
                        confidence = f"Confidence: {1 - distance:.2f}"

                        # Add label on image
                        cv2.putText(
                            img_bgr,
                            recognized_name,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2,
                        )
                    else:
                        recognition_result = "Unknown face detected"
                        confidence = (
                            f"Closest match: {face_ids[idx]} (Distance: {distance:.2f})"
                        )

            # Store pending face for learning if unknown
            global pending_face, pending_embedding
            if "Unknown" in recognition_result:
                pending_face = face_pil
                pending_embedding = embedding

            return (
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                recognition_result,
                confidence,
            )

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        return image, f"Error processing image: {str(e)}", ""


def learn_new_face(name):
    """Learn a new face and save it to the database"""
    global pending_face, pending_embedding, face_index, face_ids

    if not name.strip():
        return "Please enter a valid name", ""

    if pending_face is None or pending_embedding is None:
        return "No face detected to learn. Please upload an image first.", ""

    try:
        # Save temporary face image for learning
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            pending_face.save(tmp.name)
            tmp_path = tmp.name

        # Generate and save embeddings (now appends to existing database)
        new_embeddings = add_new_embeddings(tmp_path, name.strip())

        # Clean up temporary file
        os.unlink(tmp_path)

        # Reload the database to get the updated version
        face_index, face_ids = load_embeddings()

        # Clear pending face
        pending_face = None
        pending_embedding = None

        if new_embeddings is not None:
            return (
                f"Successfully learned and saved face for: {name.strip()}",
                f"Face database updated! Total faces: {face_index.ntotal if face_index else 0}",
            )
        else:
            return f"Failed to learn face for: {name.strip()}", ""

    except Exception as e:
        return f"Error learning new face: {str(e)}", ""


def chat_with_ai(user_input, history):
    """Chat with AI assistant using LangChain"""
    if llm is None:
        # Fallback responses when model is not available
        if "face" in user_input.lower():
            return "I can help you with face recognition! Try uploading an image and using the recognition features."
        elif "learn" in user_input.lower():
            return "To teach the system a new face, first upload an image, then go to the 'Learn New Face' tab and enter the person's name."
        else:
            return "I'm here to help with the face recognition system. Ask me about face detection, recognition, or learning new faces!"

    try:
        # Create messages for the LLM
        messages = [
            SystemMessage(
                content="You are a helpful assistant for a face recognition application. Provide helpful information about face detection, recognition, and learning new faces."
            ),
            HumanMessage(content=user_input),
        ]

        # Get response from LLM
        response = llm.invoke(messages)

        return response.content
    except Exception as e:
        return f"Error in chat: {str(e)}"


def clear_pending_face():
    """Clear the pending face for learning"""
    global pending_face, pending_embedding
    pending_face = None
    pending_embedding = None
    return "Pending face cleared. Upload a new image to continue."


# Create Gradio interface
with gr.Blocks(title="Face Recognition with Learning") as demo:
    gr.Markdown("# Face Recognition with Learning System")

    with gr.Tab("Face Recognition"):
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload Image")
                recognize_btn = gr.Button("Recognize Face", variant="primary")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Processed Image")
                recognition_result = gr.Textbox(
                    label="Recognition Result", interactive=False
                )
                confidence = gr.Textbox(label="Confidence", interactive=False)

        recognize_btn.click(
            fn=detect_and_recognize_face,
            inputs=input_image,
            outputs=[output_image, recognition_result, confidence],
        )

    with gr.Tab("Learn New Face"):
        gr.Markdown("## Teach the System a New Face")
        gr.Markdown(
            "First upload and recognize a face in the 'Face Recognition' tab, then come here to identify it."
        )

        with gr.Row():
            with gr.Column():
                person_name = gr.Textbox(
                    label="Person's Name", placeholder="Enter the person's name"
                )
                learn_btn = gr.Button("Learn Face", variant="primary")
                clear_btn = gr.Button("Clear Pending Face", variant="secondary")

            with gr.Column():
                learn_status = gr.Textbox(label="Learning Status", interactive=False)
                db_status = gr.Textbox(label="Database Status", interactive=False)

        learn_btn.click(
            fn=learn_new_face, inputs=person_name, outputs=[learn_status, db_status]
        )

        clear_btn.click(fn=clear_pending_face, outputs=[learn_status])

    with gr.Tab("AI Assistant"):
        gr.Markdown("## Chat with AI Assistant")
        gr.Markdown("Ask questions about face recognition or get help with the system.")

        chatbot = gr.Chatbot(label="Conversation", height=400)
        msg = gr.Textbox(label="Your Message", placeholder="Type your message here...")
        clear_chat = gr.Button("Clear Conversation")

        def user_input(user_message, history):
            return "", history + [[user_message, None]]

        def bot_response(history):
            if history and history[-1][1] is None:
                user_message = history[-1][0]
                bot_message = chat_with_ai(user_message, history)
                history[-1][1] = bot_message
            return history

        msg.submit(user_input, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response, chatbot, chatbot
        )

        clear_chat.click(lambda: None, outputs=chatbot)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
