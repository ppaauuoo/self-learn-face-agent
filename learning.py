import numpy as np
import faiss
from deepface import DeepFace


def generate_and_save_embeddings(
    img_path, whoami, index_file="face_embeddings.index", id_file="face_ids.npy"
):
    """Generate face embeddings using DeepFace and save them using Faiss"""

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

    # Get dimension of embeddings
    dimension = embeddings_array.shape[1]

    # Create Faiss index
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to index
    index.add(embeddings_array)

    # Save index to file
    faiss.write_index(index, index_file)

    # Save face IDs (you could associate these with actual person identifiers)
    face_ids = np.array([whoami] * len(embeddings))
    np.save(id_file, face_ids)

    print(f"Saved {len(embeddings)} embeddings to {index_file}")
    print(f"Saved face IDs to {id_file}")

    return embeddings_array


def load_embeddings(index_file="face_embeddings.index", id_file="face_ids.npy"):
    """Load embeddings from Faiss index"""
    try:
        # Load index
        index = faiss.read_index(index_file)

        # Load face IDs
        face_ids = np.load(id_file)

        print(f"Loaded {index.ntotal} embeddings from {index_file}")
        return index, face_ids
    except Exception as e:
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
