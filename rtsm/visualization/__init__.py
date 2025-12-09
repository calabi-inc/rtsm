"""
RTSM Visualization Module

Provides optional embedded visualization server for real-time point cloud
streaming and Working Memory object overlay via WebSocket.

Enable via config:
    visualization:
      enable: true
      port: 8081
"""

from rtsm.visualization.server import VisualizationServer

__all__ = ["VisualizationServer"]
