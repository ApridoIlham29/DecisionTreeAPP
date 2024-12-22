import sys
import os
import math
import time
import json
import yaml
import pickle
import logging
import threading
import queue
import signal
import numpy as np
import pandas as pd
import graphviz
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set, Any, Tuple, Union, Callable, TypeVar, Generic
from anytree import Node, RenderTree, PreOrderIter
from anytree.exporter import DotExporter
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track, Progress
from rich.panel import Panel
from rich.table import Table
from rich.style import Style
from rich.layout import Layout
from rich.text import Text
from tabulate import tabulate
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, flash
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.serving import WSGIRequestHandler
import io
import base64
import tempfile

# Type variables for generics
T = TypeVar('T')
D = TypeVar('D')

# Configure logging with rich formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("decision_tree")
console = Console()

# Custom exception hierarchy
class DecisionTreeError(Exception):
    """Base exception class for Decision Tree errors."""
    pass

class ValidationError(DecisionTreeError):
    """Raised when input validation fails."""
    pass

class VisualizationError(DecisionTreeError):
    """Raised when visualization fails."""
    pass

class DataError(DecisionTreeError):
    """Raised when there are data-related errors."""
    pass

class ConfigurationError(DecisionTreeError):
    """Raised when there are configuration errors."""
    pass

@dataclass
class DecisionTreeConfig:
    """Enhanced configuration settings for the Decision Tree Builder."""
    
    min_samples_split: int = 2
    max_depth: Optional[int] = None
    min_information_gain: float = 0.0
    
    visualization: Dict[str, Any] = field(default_factory=lambda: {
        'output_format': 'png',
        'theme': 'modern',
        'dpi': 300,
        'font_family': 'Arial',
        'show_statistics': True,
        'node_spacing': 1.2,
        'layout_direction': 'TB'
    })
    
    output: Dict[str, Any] = field(default_factory=lambda: {
        'base_dir': 'output',
        'create_timestamp_folder': True,
        'save_metadata': True,
        'save_model': True,
        'compression': True
    })
    
    logging: Dict[str, Any] = field(default_factory=lambda: {
        'level': 'INFO',
        'save_logs': True,
        'log_file': 'decision_tree.log',
        'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    })
    
    performance: Dict[str, Any] = field(default_factory=lambda: {
        'parallel_processing': False,
        'max_workers': None,
        'chunk_size': 1000,
        'cache_results': True
    })
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.min_samples_split < 2:
            raise ConfigurationError("min_samples_split must be at least 2")
        if self.max_depth is not None and self.max_depth < 1:
            raise ConfigurationError("max_depth must be at least 1")
        if not 0 <= self.min_information_gain <= 1:
            raise ConfigurationError("min_information_gain must be between 0 and 1")

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'DecisionTreeConfig':
        """Load configuration from YAML file."""
        try:
            with open(yaml_path) as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return cls()
    
    def to_yaml(self, yaml_path: Path) -> None:
        """Save configuration to YAML file."""
        try:
            with open(yaml_path, 'w') as f:
                yaml.dump(asdict(self), f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Error saving config: {e}")

class Observer(ABC):
    """Abstract base class for the Observer pattern."""
    
    @abstractmethod
    def update(self, subject: Any, *args, **kwargs) -> None:
        """Update method called by subjects."""
        pass

class Subject(ABC):
    """Abstract base class for the Subject in the Observer pattern."""
    
    def __init__(self):
        self._observers: List[Observer] = []
    
    def attach(self, observer: Observer) -> None:
        """Attach an observer."""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer: Observer) -> None:
        """Detach an observer."""
        try:
            self._observers.remove(observer)
        except ValueError:
            pass
    
    def notify(self, *args, **kwargs) -> None:
        """Notify all observers."""
        for observer in self._observers:
            observer.update(self, *args, **kwargs)

class TreeMetrics(Subject):
    """Enhanced metrics tracking for the decision tree."""
    
    def __init__(self):
        super().__init__()
        self.created_at = datetime.now()
        self.features: List[str] = []
        self.target: str = ""
        self.node_count: int = 0
        self.max_depth: int = 0
        self.leaf_count: int = 0
        self.split_counts: Dict[str, int] = {}
        self.feature_importance: Dict[str, float] = {}
        self.build_time: float = 0
        
    def update_metrics(self, node: Node) -> None:
        """Update metrics based on a node."""
        if node.name.startswith('Split:'):
            feature = node.name.split(': ')[1]
            self.split_counts[feature] = self.split_counts.get(feature, 0) + 1
        elif node.name.startswith('Leaf:'):
            self.leaf_count += 1
        self.node_count += 1
        
        # Update max depth
        depth = len(list(node.path)) - 1
        self.max_depth = max(self.max_depth, depth)
        
        self.notify('metrics_updated', node=node)
    
    def calculate_feature_importance(self) -> None:
        """Calculate feature importance based on split counts."""
        total_splits = sum(self.split_counts.values())
        if total_splits > 0:
            self.feature_importance = {
                feature: count / total_splits
                for feature, count in self.split_counts.items()
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        self.calculate_feature_importance()
        return {
            'created_at': self.created_at.isoformat(),
            'features': self.features,
            'target': self.target,
            'node_count': self.node_count,
            'max_depth': self.max_depth,
            'leaf_count': self.leaf_count,
            'split_counts': self.split_counts,
            'feature_importance': self.feature_importance,
            'build_time': round(self.build_time, 3)
        }
    
    def save(self, path: Path) -> None:
        """Save metrics to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

class TreeVisualizer(Observer):
    """Advanced decision tree visualization with clean design and improved readability."""
    
    THEMES = {
        'modern': {
            'Split': {
                'shape': 'rectangle',
                'style': 'filled,rounded',
                'fillcolor': '#3498db:#2980b9',
                'gradientangle': '90',
                'fontcolor': 'white',
                'fontname': 'Arial',
                'fontsize': '12',
                'width': '2',
                'height': '0.8',
                'penwidth': '1.5'
            },
            'Leaf': {
                'shape': 'rectangle',
                'style': 'filled,rounded',
                'fillcolor': '#2ecc71:#27ae60',
                'gradientangle': '90',
                'fontcolor': 'white',
                'fontname': 'Arial',
                'fontsize': '11',
                'width': '1.8',
                'height': '0.6',
                'penwidth': '1.5'
            },
            'Edge': {
                'color': '#34495e',
                'penwidth': '1',
                'arrowsize': '0.7',
                'arrowhead': 'vee'
            }
        }
    }
    
    def __init__(self, config: DecisionTreeConfig):
        self.config = config
        self.theme = self.THEMES['modern']
    
    def update(self, subject: Subject, *args, **kwargs) -> None:
        """Handle updates from observed subjects."""
        pass
    
    def node_style(self, node: Node) -> str:
        """Define node style for visualization."""
        if isinstance(node, Node):
            if node.name.startswith('Leaf:'):
                style = self.theme['Leaf'].copy()
                label = node.name.split(': ')[1] if ': ' in node.name else node.name
            elif node.name.startswith('Split:'):
                style = self.theme['Split'].copy()
                label = node.name.split(': ')[1] if ': ' in node.name else node.name
            else:
                return 'shape="box",style="filled",fillcolor="gray"'
            
            style['label'] = label.replace("'", "\\'")
            return ','.join(f'{k}="{v}"' for k, v in style.items())
        return 'shape="box",style="filled",fillcolor="gray"'
    
    def edge_style(self, parent: Node, child: Node) -> str:
        """Define edge style for visualization."""
        return ','.join(f'{k}="{v}"' for k, v in self.theme['Edge'].items())

class WebTreeVisualizer(TreeVisualizer):
    def create_decision_tree_graph(self, tree):
        graph = graphviz.Digraph(format='png')
        graph.attr(rankdir='TB', nodesep='0.5', ranksep='0.7', splines='polyline')
        
        def add_node(node, parent=None):
            node_id = str(id(node))
            if node.name.startswith('Split:'):
                label = node.name.split(': ')[1]
                graph.node(node_id, label, shape='ellipse', style='filled', fillcolor='#C4A484')
            elif node.name.startswith('Leaf:'):
                label = node.name.split(': ')[1]
                graph.node(node_id, label, shape='rectangle', style='filled', fillcolor='lightgreen')
            else:
                graph.node(node_id, node.name, shape='plaintext')
            
            if parent:
                graph.edge(str(id(parent)), node_id, dir='none')
            
            for child in node.children:
                add_node(child, node)
        
        add_node(tree)
        return graph

    def create_visualization_base64(self, tree: Node, metrics: Optional[TreeMetrics] = None) -> str:
        try:
            graph = self.create_decision_tree_graph(tree)
            png_data = graph.pipe(format='png')
            base64_data = base64.b64encode(png_data).decode('utf-8')
            return base64_data
        except Exception as e:
            raise VisualizationError(f"Error creating visualization: {e}")

class TreeBuilder(Subject):
    """Enhanced Decision Tree Builder with advanced features."""
    
    def __init__(self, config: Optional[DecisionTreeConfig] = None):
        super().__init__()
        self.config = config or DecisionTreeConfig()
        self.metrics = TreeMetrics()
        self.visualizer = TreeVisualizer(self.config)
        self.tree: Optional[Node] = None
        
        # Attach observers
        self.metrics.attach(self.visualizer)
    
    def entropy(self, data: pd.DataFrame, target_column: str) -> float:
        """Calculate entropy with validation and error handling."""
        try:
            if target_column not in data.columns:
                raise ValidationError(f"Target column '{target_column}' not found in data")
            
            values = data[target_column].value_counts(normalize=True)
            entropy_value = -np.sum(values * np.log2(values + np.finfo(float).eps))
            
            return entropy_value, dict(values)
            
        except Exception as e:
            logger.error(f"Error calculating entropy: {e}")
            raise
    
    def information_gain(
        self,
        data: pd.DataFrame,
        split_column: str,
        target_column: str
    ) -> float:
        """Calculate information gain with improved logging format."""
        try:
            # Calculate total entropy for the entire dataset
            total_entropy, _ = self.entropy(data, target_column)
            
            # Calculate weighted entropy for each value
            values = data[split_column].value_counts(normalize=True)
            counts = data[split_column].value_counts()
            weighted_entropy = 0
            
            self.log_message(f"Menghitung Gain untuk fitur '{split_column}':", level='info', indent=0)
            
            feature_entropies = []
            for value in values.index:
                subset = data[data[split_column] == value]
                subset_entropy, _ = self.entropy(subset, target_column)
                weight = counts[value] / len(data)
                weighted_entropy += weight * subset_entropy
                feature_entropies.append(subset_entropy)
                
                # Log value analysis
                self.log_message(f"- Nilai '{value}': entropi = {subset_entropy:.4f}", level='info', indent=1)
            
            # Calculate and log total entropy for the feature
            feature_total_entropy = sum(feature_entropies) / len(feature_entropies)
            self.log_message(f"Entropi total untuk fitur '{split_column}': {feature_total_entropy:.4f}", level='info', indent=0)
            
            # Calculate and log final gain
            gain = total_entropy - weighted_entropy
            self.log_message(f"Gain untuk fitur '{split_column}': {gain:.4f}\n", level='gain', indent=0)
            
            return gain
            
        except Exception as e:
            self.log_message(f"Error calculating information gain: {str(e)}", level='error')
            raise
    
    def best_split(self, data: pd.DataFrame, features: List[str], target_column: str) -> str:
        self.log_message("Perhitungan Gain untuk setiap fitur:", level='info', indent=0)
        
        gains = {}
        for feature in features:
            gain = self.information_gain(data, feature, target_column)
            gains[feature] = gain
        
        best_feature = max(gains, key=gains.get)
        self.log_message(f"Fitur terbaik berdasarkan Gain: {best_feature} (Gain = {gains[best_feature]:.4f})\n", level='success', indent=0)
        
        return best_feature

    def build_tree(
        self,
        data: pd.DataFrame,
        features: List[str],
        target_column: str,
        parent: Optional[Node] = None,
        depth: int = 0
    ) -> Node:
        """Build the decision tree recursively."""
        try:
            # Update metrics
            if parent is None:
                self.metrics.features = features.copy()
                self.metrics.target = target_column
                start_time = time.time()
            
            # Check stopping conditions
            if len(data[target_column].unique()) == 1:
                value = data[target_column].iloc[0]
                leaf = Node(f"Leaf: {value}", parent=parent, samples=len(data))
                self.metrics.update_metrics(leaf)
                return leaf
            
            if not features or (self.config.max_depth and depth >= self.config.max_depth):
                majority_class = data[target_column].mode()[0]
                leaf = Node(f"Leaf: {majority_class}", parent=parent, samples=len(data))
                self.metrics.update_metrics(leaf)
                return leaf
            
            # Find best split
            best_feature = self.best_split(data, features, target_column)
            root = Node(f"Split: {best_feature}", parent=parent, samples=len(data))
            self.metrics.update_metrics(root)
            
            # Create branches
            remaining_features = [f for f in features if f != best_feature]
            for value in sorted(data[best_feature].unique()):
                subset = data[data[best_feature] == value]
                if len(subset) > 0:
                    child = Node(f"{value}", parent=root)
                    self.build_tree(
                        subset,
                        remaining_features,
                        target_column,
                        parent=child,
                        depth=depth + 1
                    )
            
            # Update build time for root node
            if parent is None:
                self.metrics.build_time = time.time() - start_time
                self.tree = root
            
            return root
            
        except Exception as e:
            logger.error(f"Error building tree: {e}")
            raise

class WebTreeBuilder(TreeBuilder):
    def __init__(self, config: Optional[DecisionTreeConfig] = None):
        super().__init__(config)
        self.visualizer = WebTreeVisualizer(self.config)
        self.build_logs = []
        
    def log_message(self, message: str, level: str = 'info', indent: int = 0) -> None:
        if "Fitur terbaik berdasarkan Gain" in message:
            formatted_message = f"{'  ' * indent}⭐ {message}"
        else:
            formatted_message = f"{'  ' * indent}{message}"
        self.build_logs.append(formatted_message)
        logger.info(formatted_message)

    def information_gain(self, data: pd.DataFrame, split_column: str, target_column: str) -> float:
        try:
            total_entropy, _ = self.entropy(data, target_column)
            values = data[split_column].value_counts(normalize=True)
            counts = data[split_column].value_counts()
            weighted_entropy = 0
            
            self.log_message(f"Menghitung Gain untuk fitur '{split_column}':", level='info', indent=0)
            
            feature_entropies = []
            for value in values.index:
                subset = data[data[split_column] == value]
                subset_entropy, _ = self.entropy(subset, target_column)
                weight = counts[value] / len(data)
                weighted_entropy += weight * subset_entropy
                feature_entropies.append(subset_entropy)
                
                self.log_message(f"- Nilai '{value}': entropi = {subset_entropy:.4f}", level='info', indent=1)
            
            feature_total_entropy = sum(feature_entropies) / len(feature_entropies)
            self.log_message(f"Entropi total untuk fitur '{split_column}': {feature_total_entropy:.4f}", level='info', indent=0)
            
            gain = total_entropy - weighted_entropy
            self.log_message(f"Gain untuk fitur '{split_column}': {gain:.4f}\n", level='gain', indent=0)
            
            return gain
            
        except Exception as e:
            self.log_message(f"Error calculating information gain: {str(e)}", level='error')
            raise

    def build_tree(self, data: pd.DataFrame, features: List[str], target_column: str, 
                parent: Optional[Node] = None, depth: int = 0) -> Node:
        try:
            if parent is None:
                self.build_logs = []
                start_time = time.time()
                self.log_message("Memulai pembangunan pohon keputusan...", level='info')
                
                class_dist = data[target_column].value_counts()
                total_samples = len(data)
                self.log_message("\nDataset awal:", level='info')
                self.log_message(f"Total sampel: {total_samples}", level='info', indent=1)
                for class_name, count in class_dist.items():
                    percentage = (count / total_samples) * 100
                    self.log_message(f"Kelas '{class_name}': {count} sampel ({percentage:.1f}%)", level='info', indent=1)

            tree = super().build_tree(data, features, target_column, parent, depth)
            
            if parent is None:
                build_time = time.time() - start_time
                self.log_message(f"\nPohon keputusan selesai dibuat", level='success')
            
            return tree
            
        except Exception as e:
            self.log_message(f"Error building tree: {str(e)}", level='error')
            raise

# Create Flask application
app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# Configure SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///decision_tree_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Define DecisionTreeDataset model
class DecisionTreeDataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    parameters = db.Column(db.Text, nullable=False)
    data = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f'<DecisionTreeDataset {self.name}>'

# Create database tables
with app.app_context():
    db.create_all()

# Configure maximum request size (10MB)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Initialize global variables for web interface
current_tree_builder = None
current_visualization = None

@app.route('/')
def index():
    """Render main page."""
    datasets = DecisionTreeDataset.query.all()
    return render_template('index.html', datasets=datasets)

@app.route('/build_tree', methods=['POST'])
def build_tree():
    """Handle tree building request."""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Content-Type harus application/json'}), 400
            
        data = request.json
        if not data:
            return jsonify({'error': 'Data JSON tidak boleh kosong'}), 400
        
        # Process parameters
        parameters = [p.strip() for p in data.get('parameters', '').split(',') if p.strip()]
        if len(parameters) < 2:
            return jsonify({'error': 'Minimal harus ada 2 parameter'}), 400
            
        # Process data rows
        data_rows = [
            row.strip().split(',')
            for row in data.get('data', '').split('\n')
            if row.strip()
        ]
        
        if len(data_rows) < 2:
            return jsonify({'error': 'Minimal harus ada 2 record data'}), 400
        
        # Validate data rows
        for i, row in enumerate(data_rows, 1):
            if len(row) != len(parameters):
                return jsonify({
                    'error': f'Baris ke-{i} memiliki {len(row)} nilai, '
                            f'seharusnya {len(parameters)}'
                }), 400
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=parameters)
        features = parameters[:-1]
        target = parameters[-1]
        
        # Build tree
        global current_tree_builder
        current_tree_builder = WebTreeBuilder(DecisionTreeConfig())
        tree = current_tree_builder.build_tree(df, features, target)
        
        # Create visualization
        global current_visualization
        current_visualization = current_tree_builder.visualizer.create_visualization_base64(
            tree,
            current_tree_builder.metrics
        )
        
        # Prepare response
        response_data = {
            'success': True,
            'logs': current_tree_builder.build_logs,
            'visualization': current_visualization,
            'metrics': current_tree_builder.metrics.to_dict()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.exception("Error in build_tree endpoint:")
        return jsonify({'error': str(e)}), 500

@app.route('/save_dataset', methods=['POST'])
def save_dataset():
    """Save dataset to the database."""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Content-Type harus application/json'}), 400
            
        data = request.json
        if not data:
            return jsonify({'error': 'Data JSON tidak boleh kosong'}), 400
        
        name = data.get('name')
        parameters = data.get('parameters')
        data_rows = data.get('data')
        
        if not name or not parameters or not data_rows:
            return jsonify({'error': 'Semua field harus diisi'}), 400
        
        # Check if dataset with the same name already exists
        existing_dataset = DecisionTreeDataset.query.filter_by(name=name).first()
        if existing_dataset:
            return jsonify({'error': f'Dataset dengan nama "{name}" sudah ada'}), 400
        
        # Create new dataset
        new_dataset = DecisionTreeDataset(
            name=name,
            parameters=json.dumps(parameters),
            data=json.dumps(data_rows)
        )
        
        # Save to database
        db.session.add(new_dataset)
        db.session.commit()
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.exception("Error in save_dataset endpoint:")
        return jsonify({'error': str(e)}), 500

@app.route('/load_dataset/<int:dataset_id>')
def load_dataset(dataset_id):
    """Load dataset from the database."""
    try:
        dataset = DecisionTreeDataset.query.get(dataset_id)
        if not dataset:
            return jsonify({'error': f'Dataset with ID {dataset_id} not found'}), 404
        
        parameters = json.loads(dataset.parameters)
        data_rows = json.loads(dataset.data)
        
        return jsonify({
            'success': True,
            'name': dataset.name,
            'parameters': parameters,
            'data': data_rows
        })
        
    except Exception as e:
        logger.exception("Error in load_dataset endpoint:")
        return jsonify({'error': str(e)}), 500

@app.route('/update_dataset/<int:dataset_id>', methods=['PUT'])
def update_dataset(dataset_id):
    """Update dataset in the database."""
    try:
        dataset = DecisionTreeDataset.query.get(dataset_id)
        if not dataset:
            return jsonify({'error': f'Dataset with ID {dataset_id} not found'}), 404
        
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Content-Type harus application/json'}), 400
            
        data = request.json
        if not data:
            return jsonify({'error': 'Data JSON tidak boleh kosong'}), 400
        
        name = data.get('name')
        parameters = data.get('parameters')
        data_rows = data.get('data')
        
        if name:
            dataset.name = name
        if parameters:
            dataset.parameters = json.dumps(parameters)
        if data_rows:
            dataset.data = json.dumps(data_rows)
        
        db.session.commit()
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.exception("Error in update_dataset endpoint:")
        return jsonify({'error': str(e)}), 500

@app.route('/delete_dataset/<int:dataset_id>', methods=['DELETE'])
def delete_dataset(dataset_id):
    """Delete dataset from the database."""
    try:
        dataset = DecisionTreeDataset.query.get(dataset_id)
        if not dataset:
            return jsonify({'error': f'Dataset with ID {dataset_id} not found'}), 404
        
        db.session.delete(dataset)
        db.session.commit()
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.exception("Error in delete_dataset endpoint:")
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle large file uploads."""
    return jsonify({'error': 'File terlalu besar. Maksimal 10MB'}), 413

if __name__ == "__main__":
    try:
        # Create necessary directories
        Path('output').mkdir(exist_ok=True)
        Path('static').mkdir(exist_ok=True)
        Path('templates').mkdir(exist_ok=True)
        
        # Set the request handler to support larger content lengths
        WSGIRequestHandler.protocol_version = "HTTP/1.1"
        
        # Run Flask application
        port = 5001  # Changed from default 5000 to avoid conflicts
        print(f"\n🚀 Starting server on port {port}...")
        print(f"💻 Access the application at http://localhost:{port}")
        
        app.run(
            debug=True,
            host='0.0.0.0',
            port=port,
            threaded=True
        )
        
    except Exception as e:
        logger.exception("Failed to start application:")
        sys.exit(1)