from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {} # store (player_id : team_name) pairs

    def clustering_model(self, image):
        image_2d = image.reshape(-1, 3) 

        # KMeans with 2 clusters
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1) # k-means++ for faster convergence
        kmeans.fit(image_2d)

        return kmeans

    def player_color(self, frame, bounding_box):
        image = frame[int(bounding_box[1]) : int(bounding_box[3]), 
                      int(bounding_box[0]) : int(bounding_box[2])]
        
        top_image = image[0 : int(image.shape[0]/2), :]

        # clustering model
        kmeans = self.clustering_model(top_image)

        # get cluster labels
        labels = kmeans.labels_

        # reshape labels to image shape
        image_clustered = labels.reshape(image.shape[0], image.shape[1])

        # similar logic as jerseyAnalysis
        corner_clusters = [image_clustered[0,0], image_clustered[0, -1], image_clustered[-1, 0] , image_clustered[-1, -1]]
        background_cluster = max(set(corner_clusters), key = corner_clusters.count)
        player_cluster = 1 - background_cluster

        player_color = kmeans.cluster_centers_(player_cluster)
        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []

        for _, player_detection in player_detections.items():
            bounding_box = player_detection['bounding_box']
            player_color = self.player_color(frame, bounding_box)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        # get team color as per KMeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
    
    def get_player_team(self, frame, player_box, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.player_color(frame, player_box)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        # handling goal keeper - different jersey than other team players
        if player_id == 91:
            team_id = 1

        self.player_team_dict[player_id] = team_id

        return team_id