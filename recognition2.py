import numpy as np
import faiss
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from deepface import DeepFace
from learning import load_embeddings, search_similar_faces


def plot_faces_with_matches(query_image_path, results, top_k=3):
    """Plot the original face with its top matches"""
    
    try:
        # Load the original image
        img = plt.imread(query_image_path)
        
        # Create subplots
        fig, axes = plt.subplots(len(results), top_k + 1, figsize=(15, 5 * len(results)))
        if len(results) == 1:
            axes = axes.reshape(1, -1)
        
        for i, face_result in enumerate(results):
            # Get face detection results for bounding boxes
            embedding_objs = DeepFace.represent(img_path=query_image_path)
            
            # Plot original face
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Original Face #{face_result['face_number']}")
            axes[i, 0].axis('off')
            
            # Add bounding box for detected face
            if i < len(embedding_objs) and 'facial_area' in embedding_objs[i]:
                facial_area = embedding_objs[i]['facial_area']
                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                       edgecolor='red', facecolor='none')
                axes[i, 0].add_patch(rect)
            
            # Plot similar faces
            for j in range(min(top_k, len(face_result['matches']))):
                match = face_result['matches'][j]
                
                # For now, we'll create placeholder text since we don't have 
                # the actual face images from the face_ids
                axes[i, j + 1].text(0.5, 0.5, 
                                   f"Match #{match['rank']}\nID: {match['face_id']}\nDist: {match['distance']:.4f}",
                                   ha='center', va='center', fontsize=12)
                axes[i, j + 1].set_title(f"Similar Face #{match['rank']}")
                axes[i, j + 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error plotting faces: {e}")


def recognize_faces(query_image_path, index_file="face_embeddings.index", id_file="face_ids.npy", k=5):
    """Recognize faces in query image using FAISS index"""
    
    # Load pre-built FAISS index
    index, face_ids = load_embeddings(index_file, id_file)
    
    if index is None:
        print("No FAISS index found. Please run learning.py first to generate embeddings.")
        return None
    
    # Generate embeddings for query image
    try:
        embedding_objs = DeepFace.represent(img_path=query_image_path)
        
        if not embedding_objs:
            print("No faces detected in the query image")
            return None
        
        # Process each detected face
        results = []
        for i, obj in enumerate(embedding_objs):
            query_embedding = np.array(obj["embedding"], dtype=np.float32)
            
            # Search for similar faces
            distances, indices = search_similar_faces(query_embedding, index, k)
            
            face_result = {
                "face_number": i,
                "matches": []
            }
            
            for j, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                face_result["matches"].append({
                    "rank": j + 1,
                    "face_id": face_ids[idx],
                    "distance": float(dist)
                })
            
            results.append(face_result)
        
        return results
        
    except Exception as e:
        print(f"Error during face recognition: {e}")
        return None


def main():
    # Example usage
    query_image = "img.png"
    results = recognize_faces(query_image)
    
    if results:
        print(f"Face recognition results for {query_image}:")
        for face in results:
            print(f"\nFace #{face['face_number']}:")
            for match in face["matches"]:
                print(f"  {match['rank']}. {match['face_id']} (distance: {match['distance']:.4f})")
        
        # Plot the results
        print("\nPlotting faces with matches...")
        plot_faces_with_matches(query_image, results, top_k=3)
    else:
        print("No recognition results found.")


if __name__ == "__main__":
    main()
