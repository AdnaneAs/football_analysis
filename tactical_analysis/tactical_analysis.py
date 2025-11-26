import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class TacticalAnalysis:
    def __init__(self):
        # Pitch dimensions in meters (based on ViewTransformer)
        self.width = 68
        self.length = 23.32 # This is the transformed section length
        
        # Visualization settings
        self.pitch_color = (50, 168, 82) # Green
        self.line_color = (255, 255, 255) # White
        self.scale = 10 # Pixels per meter for visualization

    def draw_pitch(self, width, height):
        """
        Draws a simple 2D pitch background.
        """
        pitch_img = np.ones((int(height * self.scale), int(width * self.scale), 3), dtype=np.uint8) * 255
        pitch_img[:] = self.pitch_color
        
        # Draw boundaries
        cv2.rectangle(pitch_img, (0, 0), (int(width * self.scale), int(height * self.scale)), self.line_color, 2)
        
        return pitch_img

    def draw_voronoi(self, frames_tracks):
        """
        Generates a video of Voronoi diagrams for each frame.
        """
        output_frames = []
        
        for frame_num, player_tracks in enumerate(frames_tracks['players']):
            # Create blank pitch
            pitch_img = self.draw_pitch(self.length, self.width)
            
            points = []
            colors = []
            
            # Collect player positions and team colors
            for player_id, track in player_tracks.items():
                if 'position_transformed' in track and track['position_transformed'] is not None:
                    pos = track['position_transformed']
                    # Check if position is within bounds
                    if 0 <= pos[0] <= self.length and 0 <= pos[1] <= self.width:
                        points.append([pos[0] * self.scale, pos[1] * self.scale])
                        
                        # Get team color (BGR)
                        team_color = track.get('team_color', (255, 255, 255))
                        colors.append(team_color)

            if len(points) > 3: # Voronoi needs at least 4 points to be interesting/stable
                points = np.array(points)
                
                # Add dummy points far away to bound the regions (optional but helps with edge cases)
                # For simplicity, we'll just use the points we have.
                
                try:
                    vor = Voronoi(points)
                    
                    # Draw regions
                    # Note: Drawing Voronoi regions manually in OpenCV is complex because regions can be infinite.
                    # A simpler approach for visualization is to iterate over every pixel (slow) or use matplotlib.
                    # Here we will use a subdivision approach or nearest neighbor for speed.
                    
                    # Fast Approximate Voronoi:
                    # Create a small map, fill with player indices, resize up.
                    
                    h, w, _ = pitch_img.shape
                    subdiv = cv2.Subdiv2D((0, 0, w, h))
                    
                    for p in points:
                        # Clamp points to be strictly inside to avoid Subdiv2D errors
                        x = max(1, min(w - 2, p[0]))
                        y = max(1, min(h - 2, p[1]))
                        subdiv.insert((float(x), float(y)))
                        
                    (facets, centers) = subdiv.getVoronoiFacetList([])
                    
                    for i in range(len(facets)):
                        facet = facets[i]
                        center = centers[i]
                        
                        # Find which player this center belongs to
                        # (The order of centers returned by getVoronoiFacetList might not match input order)
                        # So we find the closest original point to this center
                        min_dist = float('inf')
                        idx = -1
                        for j, p in enumerate(points):
                            dist = np.linalg.norm(p - center)
                            if dist < min_dist:
                                min_dist = dist
                                idx = j
                        
                        if idx != -1:
                            color = colors[idx]
                            # Convert float polygon to int
                            facet_poly = np.array(facet, dtype=np.int32)
                            
                            # Draw polygon with semi-transparency
                            overlay = pitch_img.copy()
                            cv2.fillPoly(overlay, [facet_poly], color)
                            cv2.addWeighted(overlay, 0.4, pitch_img, 0.6, 0, pitch_img)
                            
                            # Draw player dot
                            cv2.circle(pitch_img, (int(points[idx][0]), int(points[idx][1])), 5, (0,0,0), -1)
                            cv2.circle(pitch_img, (int(points[idx][0]), int(points[idx][1])), 3, color, -1)

                except Exception as e:
                    print(f"Voronoi error frame {frame_num}: {e}")
            
            output_frames.append(pitch_img)
            
        return output_frames

    def draw_heatmap(self, tracks, player_id):
        """
        Generates a heatmap for a specific player.
        """
        positions = []
        for frame_tracks in tracks['players']:
            if player_id in frame_tracks:
                track = frame_tracks[player_id]
                if 'position_transformed' in track and track['position_transformed'] is not None:
                    pos = track['position_transformed']
                    # Swap x and y for plotting if needed, depending on orientation
                    # Here we assume x is length (0-23) and y is width (0-68)
                    positions.append(pos)
        
        if not positions:
            return None
            
        positions = np.array(positions)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 12))
        
        # Draw pitch background
        ax.set_facecolor('#32a852')
        ax.set_xlim(0, self.length)
        ax.set_ylim(0, self.width)
        
        # Create KDE Plot (Heatmap)
        import seaborn as sns
        try:
            sns.kdeplot(x=positions[:,0], y=positions[:,1], fill=True, thresh=0.05, alpha=0.5, cmap='hot', ax=ax)
        except ImportError:
            print("Seaborn not installed, using scatter plot")
            ax.scatter(positions[:,0], positions[:,1], c='red', alpha=0.1)
            
        ax.set_title(f'Heatmap for Player {player_id}')
        ax.invert_yaxis() # Match image coordinate system if needed
        
        # Convert plot to image
        canvas = FigureCanvas(fig)
        canvas.draw()
        
        # Use buffer_rgba() instead of tostring_rgb() which is deprecated/removed
        img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        img = img.reshape(canvas.get_width_height()[::-1] + (4,))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        plt.close(fig)
        return img
