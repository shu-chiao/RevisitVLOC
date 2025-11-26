"""
VPR Database Manager - Store and query NetVLAD features using Milvus
"""

import numpy as np
import h5py
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from typing import List
from pathlib import Path

# DB Schema (place_memory):
# | image_id | image_name              | vertex_id | pose_x | pose_y | descriptor    |
# |----------|-------------------------|-----------|--------|--------|---------------|
# | 1        | vertex_005_000034.png   | 5         | 0.0    | 0.0    | [0.12, ...]   |
# | 2        | vertex_006_000035.png   | 6         | 0.0    | 0.0    | [0.08, ...]   |
#
# Search returns: [(image_name, vertex_id, pose_x, pose_y, similarity), ...]



class NetVLADMilvusDB:
    """Milvus manager for NetVLAD image-level features with pose metadata"""

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

    def create_collection(self, dim=4096, drop_old=False):
        """
        Create Milvus collection for NetVLAD reference images with pose metadata

        Args:
            dim: Dimension of NetVLAD descriptor (default 4096)
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

        # Define schema: image_name, vertex_id, pose_x, pose_y, descriptor
        fields = [
            FieldSchema(name="image_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="image_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="vertex_id", dtype=DataType.INT64),
            FieldSchema(name="pose_x", dtype=DataType.FLOAT),
            FieldSchema(name="pose_y", dtype=DataType.FLOAT),
            FieldSchema(name="descriptor", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]

        schema = CollectionSchema(fields, description="NetVLAD Reference Images with Pose")

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

        Returns:
            Number of inserted records
        """
        if self.collection is None:
            raise ValueError("Collection not created. Call create_collection() first.")

        print(f"\nLoading features from {h5_path}...")

        # Load descriptors from HDF5
        image_names = []
        descriptors = []
        vertex_ids = []
        poses_x = []
        poses_y = []

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
                poses_x.append(0.0)  # Default pose (will be from SLAM in production)
                poses_y.append(0.0)

        print(f"Loaded {len(image_names)} images")

        # Insert in batches
        batch_size = 1000
        total_inserted = 0

        for i in range(0, len(image_names), batch_size):
            batch_names = image_names[i:i+batch_size]
            batch_vertices = vertex_ids[i:i+batch_size]
            batch_poses_x = poses_x[i:i+batch_size]
            batch_poses_y = poses_y[i:i+batch_size]
            batch_descs = descriptors[i:i+batch_size]

            entities = [
                batch_names,
                batch_vertices,
                batch_poses_x,
                batch_poses_y,
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

    def insert_row(self, image_name: str, vertex_id: int, pose_x: float, pose_y: float, descriptor: np.ndarray):
        """
        Insert a single row into the database

        Args:
            image_name: Image filename
            vertex_id: SLAM vertex ID
            pose_x: X coordinate (meters)
            pose_y: Y coordinate (meters)
            descriptor: NetVLAD descriptor vector (assumed already normalized from h5)
        """
        if self.collection is None:
            raise ValueError("Collection not created. Call create_collection() first.")

        entities = [
            [image_name],
            [vertex_id],
            [pose_x],
            [pose_y],
            [descriptor.tolist()]
        ]

        self.collection.insert(entities)
        self.collection.flush()

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List:
        """
        Search for similar images in Milvus

        Args:
            query_vector: Query descriptor array (will be normalized internally)
            top_k: Number of top matches to return

        Returns:
            List of tuples: [(image_name, vertex_id, pose_x, pose_y, similarity), ...]
        """
        if self.collection is None:
            self.collection = Collection(self.collection_name)

        # Load collection
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
            output_fields=["image_name", "vertex_id", "pose_x", "pose_y"]
        )

        # Format results
        matches = []
        for hit in results[0]:
            match_name = hit.entity.get('image_name')
            vertex_id = hit.entity.get('vertex_id')
            pose_x = hit.entity.get('pose_x')
            pose_y = hit.entity.get('pose_y')
            similarity = hit.distance  # IP distance (higher = more similar)
            matches.append((match_name, vertex_id, pose_x, pose_y, float(similarity)))

        return matches

    def close(self):
        """Close Milvus connection"""
        connections.disconnect("default")
        print("✓ Disconnected from Milvus")
