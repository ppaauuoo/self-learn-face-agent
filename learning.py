import numpy as np
import faiss
from deepface import DeepFace


def add_new_embeddings(
    img_path, whoami, index_file="face_embeddings.index", id_file="face_ids.npy"
):
    """Generate face embeddings and append them to existing Faiss database"""
    
    # Generate embeddings
    embedding_objs = DeepFace.represent(img_path=img_path)

    # Extract embeddings and ensure they're numpy arrays
    embeddings = []
    for obj in embedding_objs:
        embedding = np.array(obj["embedding"], dtype=np.float32)
        embeddings.append(embedding)

    if not embeddings:
        print("No faces detected in the image")
        return None

    # Convert to numpy array
    embeddings_array = np.array(embeddings)
    
    try:
        # Load existing index and face IDs
        index, face_ids = load_embeddings(index_file, id_file)
        
        if index is None or face_ids is None:
            # If no existing database, create new one
            print("No existing database found, creating new one...")
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatL2(dimension)
            face_ids = np.array([])
        
        # Add new embeddings to existing index
        index.add(embeddings_array)
        
        # Add new face IDs
        new_face_ids = np.array([whoami] * len(embeddings))
        face_ids = np.concatenate([face_ids, new_face_ids])
        
        # Save updated index and face IDs
        faiss.write_index(index, index_file)
        np.save(id_file, face_ids)
        
        print(f"Added {len(embeddings)} new embeddings for {whoami}")
        print(f"Total embeddings in database: {index.ntotal}")
        
        return embeddings_array
        
    except Exception as e:
        print(f"Error adding new embeddings: {e}")
        return None


def generate_and_save_embeddings(
    img_path, whoami, index_file="face_embeddings.index", id_file="face_ids.npy"
):
    """Generate face embeddings using DeepFace and save them using Faiss"""
    # This function now uses the new append functionality
    return add_new_embeddings(img_path, whoami, index_file, id_file)


def load_embeddings(index_file="face_embeddings.index", id_file="face_ids.npy"):
    """Load embeddings from Faiss index"""
    try:
        # Load index
        index = faiss.read_index(index_file)

        # Load face IDs
        face_ids = np.load(id_file)

        print(f"Loaded {index.ntotal} embeddings from {index_file}")
        return index, face_ids
    except (FileNotFoundError, Exception) as e:
        print(f"Error loading embeddings: {e}")
        return None, None


def search_similar_faces(query_embedding, index, k=5):
    """Search for similar faces in the Faiss index"""
    if index is None:
        print("No index available for search")
        return None

    # Ensure query embedding is the right format
    if not isinstance(query_embedding, np.ndarray):
        query_embedding = np.array(query_embedding, dtype=np.float32)

    # Reshape if needed
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Search
    distances, indices = index.search(query_embedding, k)

    return distances, indices


# Example usage
if __name__ == "__main__":
    # Generate and save embeddings
    embeddings = generate_and_save_embeddings("img.png", "Opal")

    if embeddings is not None:
        # Load embeddings back
        index, face_ids = load_embeddings()

        # Search for similar faces (using first embedding as query)
        if index is not None:
            query = embeddings[0]
            distances, indices = search_similar_faces(query, index)

            print("Search results:")
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                print(f"  Result {i + 1}: {face_ids[idx]}, Distance: {dist:.4f}")
