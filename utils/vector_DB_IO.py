"""
VPR Database Manager - Store and query reference database using Milvus
"""

import os
import pickle
import numpy as np
import torch
import h5py
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from typing import List, Dict, Tuple
from pathlib import Path

try:
    from place_rec_global_config import datasets, experiments, workdir_data
except ImportError:
    # Allow module to work without this config file
    datasets, experiments, workdir_data = None, None, None


class VPRDatabaseManager:
    def __init__(self, db_path="./milvus_vpr.db", collection_name="reference_segments"):
        """
        Initialize Milvus connection and collection (using Milvus Lite - local mode)

        Args:
            db_path: Path to local Milvus database file (for Milvus Lite)
            collection_name: Name of the collection to use
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.collection = None

        # Connect to Milvus Lite (local embedded mode)
        connections.connect("default", uri=db_path)
        print(f"Connected to Milvus Lite at {db_path}")

    def create_collection(self, dim=1024, drop_old=False):
        """
        Create Milvus collection for reference segments

        Args:
            dim: Dimension of descriptor vectors (1024 after PCA)
            drop_old: Whether to drop existing collection
        """
        # Drop old collection if exists
        if drop_old and utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"Dropped existing collection: {self.collection_name}")

        # Define schema
        fields = [
            FieldSchema(name="segment_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="image_id", dtype=DataType.INT64),
            FieldSchema(name="image_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="descriptor", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]

        schema = CollectionSchema(fields, description="VPR Reference Segments")

        # Create collection
        self.collection = Collection(self.collection_name, schema)
        print(f"Created collection: {self.collection_name}")

        # Create index for vector field
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index("descriptor", index_params)
        print("Created index on descriptor field")

        return self.collection

    def load_reference_db(self, dataset: str, experiment_name: str, domain: str, results_pkl_suffix: str) -> Tuple:
        """
        Load reference database files from disk

        Args:
            dataset: Dataset name (e.g., 'tornare2')
            experiment_name: Experiment folder name (e.g., 'exp0_global_SegLoc_VLAD_PCA_o3_tornare2_01102025_165827')
            domain: VLAD domain (e.g., 'indoor')
            results_pkl_suffix: Suffix from experiment config (e.g., '_results_exp11_global_SegLoc_VLAD_PCA_o3.pkl')

        Returns:
            (segFtVLAD1, imInds1, segRange1, image_names)
        """
        workdir = f'{workdir_data}/{dataset}/out'
        results_dir = f"{workdir}/results/global/{experiment_name}"

        # Load segFtVLAD1
        segFtVLAD1_file = f"{results_dir}/{dataset}_segFtVLAD1_domain_{domain}__{results_pkl_suffix}"
        with open(segFtVLAD1_file, 'rb') as f:
            segFtVLAD1 = pickle.load(f)
        print(f"Loaded segFtVLAD1: {segFtVLAD1.shape}")

        # Load imInds1
        imInds1_file = f"{results_dir}/{dataset}_imInds1_domain_{domain}__{results_pkl_suffix}"
        with open(imInds1_file, 'rb') as f:
            imInds1 = pickle.load(f)
        print(f"Loaded imInds1: {len(imInds1)} segments")

        # Load segRange1
        segRange1_file = f"{results_dir}/{dataset}_segRange1_domain_{domain}__{results_pkl_suffix}"
        with open(segRange1_file, 'rb') as f:
            segRange1 = pickle.load(f)
        print(f"Loaded segRange1: {len(segRange1)} images")

        # Load reference image names
        dataset_config = datasets[dataset]
        dataPath_r = f"{workdir_data}/{dataset}/{dataset_config['data_subpath1_r']}/"
        image_files = sorted([f for f in os.listdir(dataPath_r) if f.endswith(('.jpg', '.png', '.jpeg'))])
        print(f"Loaded {len(image_files)} reference image names")

        return segFtVLAD1, imInds1, segRange1, image_files

    def load_from_directory(self, results_dir: str, image_dir: str) -> Tuple:
        """
        Auto-discover and load reference database files from directory

        Args:
            results_dir: Directory containing segFtVLAD1, imInds1, segRange1 pickle files
            image_dir: Directory containing reference images

        Returns:
            (segFtVLAD1, imInds1, segRange1, image_names)
        """
        # Find files by pattern matching
        all_files = os.listdir(results_dir)

        # Find segFtVLAD1 file
        segFtVLAD1_files = [f for f in all_files if 'segFtVLAD' in f and f.endswith('.pkl')]
        if not segFtVLAD1_files:
            raise FileNotFoundError(f"No segFtVLAD file found in {results_dir}")
        segFtVLAD1_file = os.path.join(results_dir, segFtVLAD1_files[0])

        # Find imInds1 file
        imInds1_files = [f for f in all_files if 'imInds' in f and f.endswith('.pkl')]
        if not imInds1_files:
            raise FileNotFoundError(f"No imInds file found in {results_dir}")
        imInds1_file = os.path.join(results_dir, imInds1_files[0])

        # Find segRange1 file
        segRange1_files = [f for f in all_files if 'segRange' in f and f.endswith('.pkl')]
        if not segRange1_files:
            raise FileNotFoundError(f"No segRange file found in {results_dir}")
        segRange1_file = os.path.join(results_dir, segRange1_files[0])

        # Load files
        print(f"Loading {segFtVLAD1_files[0]}...")
        with open(segFtVLAD1_file, 'rb') as f:
            segFtVLAD1 = pickle.load(f)
        print(f"Loaded segFtVLAD1: {segFtVLAD1.shape}")

        print(f"Loading {imInds1_files[0]}...")
        with open(imInds1_file, 'rb') as f:
            imInds1 = pickle.load(f)
        print(f"Loaded imInds1: {len(imInds1)} segments")

        print(f"Loading {segRange1_files[0]}...")
        with open(segRange1_file, 'rb') as f:
            segRange1 = pickle.load(f)
        print(f"Loaded segRange1: {len(segRange1)} images")

        # Load reference image names
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        print(f"Loaded {len(image_files)} reference image names from {image_dir}")

        return segFtVLAD1, imInds1, segRange1, image_files

    def insert_reference_data(self, segFtVLAD1: torch.Tensor, imInds1: np.ndarray, image_names: List[str]):
        """
        Insert reference segments into Milvus

        Args:
            segFtVLAD1: Reference descriptors tensor [N_segments, dim]
            imInds1: Segment-to-image mapping array [N_segments]
            image_names: List of reference image filenames
        """
        if self.collection is None:
            raise ValueError("Collection not created. Call create_collection() first.")

        # Convert tensor to numpy
        if isinstance(segFtVLAD1, torch.Tensor):
            descriptors = segFtVLAD1.cpu().numpy()
        else:
            descriptors = segFtVLAD1

        # Prepare data for insertion
        segment_ids = list(range(len(descriptors)))
        image_ids = imInds1.tolist()
        image_name_list = [image_names[img_id] for img_id in image_ids]
        descriptor_list = descriptors.tolist()

        # Insert in batches
        batch_size = 10000
        total_inserted = 0

        for i in range(0, len(segment_ids), batch_size):
            batch_segment_ids = segment_ids[i:i+batch_size]
            batch_image_ids = image_ids[i:i+batch_size]
            batch_image_names = image_name_list[i:i+batch_size]
            batch_descriptors = descriptor_list[i:i+batch_size]

            entities = [
                batch_segment_ids,
                batch_image_ids,
                batch_image_names,
                batch_descriptors
            ]

            self.collection.insert(entities)
            total_inserted += len(batch_segment_ids)
            print(f"Inserted {total_inserted}/{len(segment_ids)} segments")

        # Flush to persist data
        self.collection.flush()
        print(f"Successfully inserted {total_inserted} segments into Milvus")

        # Load collection for searching
        self.collection.load()
        print("Collection loaded and ready for search")

    def search_segments(self, query_descriptors: np.ndarray, top_k: int = 50) -> List[Dict]:
        """
        Search for similar segments in Milvus

        Args:
            query_descriptors: Query descriptor vectors [N_query_segments, dim]
            top_k: Number of top matches per query segment

        Returns:
            List of search results
        """
        if self.collection is None:
            self.collection = Collection(self.collection_name)
            self.collection.load()

        # Ensure collection is loaded
        if not self.collection.is_loaded:
            self.collection.load()

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        results = self.collection.search(
            data=query_descriptors.tolist(),
            anns_field="descriptor",
            param=search_params,
            limit=top_k,
            output_fields=["segment_id", "image_id", "image_name"]
        )

        return results

    def aggregate_to_images(self, search_results, top_n: int = 5) -> List[Tuple[int, str, int, float]]:
        """
        Aggregate segment matches to image-level results using voting

        Args:
            search_results: Results from search_segments()
            top_n: Number of top images to return

        Returns:
            List of (image_id, image_name, vote_count, best_similarity)
        """
        # Count votes per image
        image_votes = {}

        for query_result in search_results:
            for hit in query_result:
                image_id = hit.entity.get('image_id')
                image_name = hit.entity.get('image_name')
                distance = hit.distance

                if image_id not in image_votes:
                    image_votes[image_id] = {
                        'image_name': image_name,
                        'votes': 0,
                        'best_distance': float('inf')
                    }

                image_votes[image_id]['votes'] += 1
                image_votes[image_id]['best_distance'] = min(
                    image_votes[image_id]['best_distance'],
                    distance
                )

        # Sort by votes (descending) then by best distance (ascending)
        sorted_images = sorted(
            image_votes.items(),
            key=lambda x: (-x[1]['votes'], x[1]['best_distance'])
        )

        # Return top N
        results = []
        for image_id, info in sorted_images[:top_n]:
            # Convert L2 distance to similarity score [0, 1]
            similarity = (2 - info['best_distance']) / 2
            results.append((
                image_id,
                info['image_name'],
                info['votes'],
                similarity
            ))

        return results

    def save_metadata(self, pca_model_path: str, config: Dict, output_path: str):
        """
        Save metadata (PCA model path and config) to file

        Args:
            pca_model_path: Path to PCA model file
            config: Configuration dictionary
            output_path: Where to save metadata
        """
        metadata = {
            'pca_model_path': pca_model_path,
            'config': config,
            'collection_name': self.collection_name
        }

        with open(output_path, 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Saved metadata to {output_path}")

    def close(self):
        """Close Milvus connection"""
        connections.disconnect("default")
        print("Disconnected from Milvus")


class NetVLADMilvusDB:
    """Simplified Milvus manager for NetVLAD image-level features"""

    def __init__(self, db_path, collection_name="netvlad_references"):
        """
        Initialize Milvus connection (Milvus Lite - local embedded mode)

        Args:
            db_path: Path to local Milvus database file
            collection_name: Name of the collection
        """
        self.db_path = str(db_path)
        self.collection_name = collection_name
        self.collection = None

        # Connect to Milvus Lite
        connections.connect("default", uri=self.db_path)
        print(f"✓ Connected to Milvus Lite at {self.db_path}")

    def create_collection(self, dim=32768, drop_old=False):
        """
        Create Milvus collection for NetVLAD reference images

        Args:
            dim: Dimension of NetVLAD descriptor (default 32768 for netvlad)
            drop_old: Whether to drop existing collection
        """
        # Drop old collection if exists
        if drop_old and utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"✓ Dropped existing collection: {self.collection_name}")

        # Check if collection already exists
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            print(f"✓ Loaded existing collection: {self.collection_name}")
            return self.collection

        # Define schema: image_id, image_name, vertex_id, descriptor
        fields = [
            FieldSchema(name="image_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="image_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="vertex_id", dtype=DataType.INT64),
            FieldSchema(name="descriptor", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]

        schema = CollectionSchema(fields, description="NetVLAD Reference Images")

        # Create collection
        self.collection = Collection(self.collection_name, schema)
        print(f"✓ Created collection: {self.collection_name}")

        # Create index for vector field
        index_params = {
            "metric_type": "IP",  # Inner Product (cosine similarity for normalized vectors)
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index("descriptor", index_params)
        print("✓ Created index on descriptor field")

        return self.collection

    def insert_from_h5(self, h5_path: Path, vertex_mapping: dict = None):
        """
        Insert reference features from HDF5 file into Milvus

        Args:
            h5_path: Path to NetVLAD features HDF5 file
            vertex_mapping: Dict mapping image_name -> vertex_id (optional)
        """
        if self.collection is None:
            raise ValueError("Collection not created. Call create_collection() first.")

        print(f"\nLoading features from {h5_path}...")

        # Load descriptors from HDF5
        image_names = []
        descriptors = []
        vertex_ids = []

        with h5py.File(h5_path, 'r') as f:
            for img_name in f.keys():
                desc = f[img_name]['global_descriptor'][:]

                # Normalize descriptor for cosine similarity
                desc = desc / np.linalg.norm(desc)

                image_names.append(img_name)
                descriptors.append(desc.tolist())

                # Extract vertex ID from filename if available
                if vertex_mapping and img_name in vertex_mapping:
                    v_id = vertex_mapping[img_name]
                else:
                    # Try to parse from filename: vertex_005_... -> 5
                    parts = img_name.split('_')
                    if parts[0] == 'vertex' and len(parts) >= 2:
                        v_id = int(parts[1])
                    else:
                        v_id = -1  # Unknown vertex

                vertex_ids.append(v_id)

        print(f"Loaded {len(image_names)} images")

        # Insert in batches
        batch_size = 1000
        total_inserted = 0

        for i in range(0, len(image_names), batch_size):
            batch_names = image_names[i:i+batch_size]
            batch_vertices = vertex_ids[i:i+batch_size]
            batch_descs = descriptors[i:i+batch_size]

            entities = [
                batch_names,
                batch_vertices,
                batch_descs
            ]

            self.collection.insert(entities)
            total_inserted += len(batch_names)
            print(f"  Inserted {total_inserted}/{len(image_names)} images")

        # Flush to persist data
        self.collection.flush()
        print(f"✓ Successfully inserted {total_inserted} images into Milvus")

        # Load collection for searching
        self.collection.load()
        print("✓ Collection loaded and ready for search")

        return total_inserted

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        """
        Search for similar images in Milvus with a single query vector

        Args:
            query_vector: Query descriptor array (will be normalized internally)
            top_k: Number of top matches to return

        Returns:
            List of tuples: [(match_name, vertex_id, similarity_score), ...]
        """
        if self.collection is None:
            self.collection = Collection(self.collection_name)

        # Load collection (idempotent - safe to call multiple times)
        self.collection.load()

        # Normalize query vector for cosine similarity
        query_vector = query_vector / np.linalg.norm(query_vector)

        # Search parameters
        search_params = {
            "metric_type": "IP",  # Inner Product (cosine similarity)
            "params": {"nprobe": 10}
        }

        results = self.collection.search(
            data=[query_vector.tolist()],
            anns_field="descriptor",
            param=search_params,
            limit=top_k,
            output_fields=["image_name", "vertex_id"]
        )

        # Format results
        matches = []
        for hit in results[0]:
            match_name = hit.entity.get('image_name')
            vertex_id = hit.entity.get('vertex_id')
            similarity = hit.distance  # IP distance (higher = more similar)
            matches.append((match_name, vertex_id, float(similarity)))

        return matches

    def search_batch(self, query_descriptors: dict, top_k: int = 5):
        """
        Search for similar images in Milvus with multiple query vectors

        Args:
            query_descriptors: Dict {image_name: descriptor_array}
            top_k: Number of top matches per query

        Returns:
            Dict {query_image_name: [(match_name, vertex_id, similarity_score), ...]}
        """
        if self.collection is None:
            self.collection = Collection(self.collection_name)

        # Load collection (idempotent - safe to call multiple times)
        self.collection.load()

        # Prepare query data
        query_names = list(query_descriptors.keys())
        query_vectors = []

        for name in query_names:
            desc = query_descriptors[name]
            # Normalize for cosine similarity
            desc = desc / np.linalg.norm(desc)
            query_vectors.append(desc.tolist())

        # Search parameters
        search_params = {
            "metric_type": "IP",  # Inner Product (cosine similarity)
            "params": {"nprobe": 10}
        }

        results = self.collection.search(
            data=query_vectors,
            anns_field="descriptor",
            param=search_params,
            limit=top_k,
            output_fields=["image_name", "vertex_id"]
        )

        # Format results
        formatted_results = {}
        for query_name, result in zip(query_names, results):
            matches = []
            for hit in result:
                match_name = hit.entity.get('image_name')
                vertex_id = hit.entity.get('vertex_id')
                similarity = hit.distance  # IP distance (higher = more similar)
                matches.append((match_name, vertex_id, float(similarity)))

            formatted_results[query_name] = matches

        return formatted_results

    def close(self):
        """Close Milvus connection"""
        connections.disconnect("default")
        print("✓ Disconnected from Milvus")


# Example usage
if __name__ == "__main__":
    # Initialize manager (using Milvus Lite - local mode)
    manager = VPRDatabaseManager(db_path="./milvus_vpr.db")

    # Create collection
    manager.create_collection(dim=1024, drop_old=True)

    # Load reference DB
    dataset = "tornare2"
    experiment_name = "exp0_global_SegLoc_VLAD_PCA_o3_tornare2_02102025_151351"
    domain = "indoor"
    results_pkl_suffix = "_results_exp11_global_SegLoc_VLAD_PCA_o3.pkl"

    segFtVLAD1, imInds1, segRange1, image_names = manager.load_reference_db(
        dataset, experiment_name, domain, results_pkl_suffix
    )

    # Insert into Milvus
    manager.insert_reference_data(segFtVLAD1, imInds1, image_names)

    # Save metadata
    pca_model_path = f"{workdir_data}/{dataset}/out/{dataset}_r_fitted_pca_model_order3.pkl"
    config = {
        'dataset': dataset,
        'experiment': 'exp0_global_SegLoc_VLAD_PCA_o3',
        'vocab_vlad': 'domain',
        'order': 3,
        'domain': domain
    }
    manager.save_metadata(pca_model_path, config, f"{workdir_data}/{dataset}/out/milvus_metadata.pkl")

    # Close connection
    manager.close()
