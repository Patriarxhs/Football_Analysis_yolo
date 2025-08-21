from sklearn.cluster import KMeans

class Team_Assigner:
    def __init__(self):
        self.team_colours = {}
        self.player_team={}
        pass
    
    
    def get_clustering_model(self, image):
        image_2d=image.reshape(-1,3)

        #k-means with 2 clusters
        k_means=KMeans(n_clusters=2,init='k-means++',n_init=10,random_state=0).fit(image_2d)
        return k_means

        
    
    def get_player_colour(self,frame, bbox):
        image=frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        top_half_of_image=image[0:int(image.shape[0]/2),:]
        
        k_means=self.get_clustering_model(top_half_of_image)
        
        #get clusters
        labels=k_means.labels_

        #reshape
        clustered_image=labels.reshape(top_half_of_image.shape[0],top_half_of_image.shape[1])
        
        corner_clusters=[clustered_image[0,0],clustered_image[0,-1],
                 clustered_image[-1,0],clustered_image[-1,-1]]

        non_player=max(set(corner_clusters),key=corner_clusters.count)
        player_cluster=1-non_player
        
        player_colour=k_means.cluster_centers_[player_cluster]
        
        return player_colour
    
    def assign_team_colour(self,frame,player_detections):
        player_colours=[]
        
        for _,player_detection in player_detections.items():
            bbox=player_detection['bbox']
            player_colour=self.get_player_colour(frame, bbox)
            player_colours.append(player_colour)
        
        kmeans=KMeans(n_clusters=2, init='k-means++', n_init=10).fit(player_colours)
        self.kmeans=kmeans
        self.team_colours[1] = kmeans.cluster_centers_[0]
        self.team_colours[2] = kmeans.cluster_centers_[1]
        
        
    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team:
            return self.player_team[player_id]
        
        player_colour=self.get_player_colour(frame, player_bbox)
        
        team_id=self.kmeans.predict(player_colour.reshape(1,-1))[0]
        team_id+=1 # to make it 1 or 2 instead of 0 or 1
        if team_id==91:
            team_id=1
        self.player_team[player_id]=team_id
        
        
        return team_id
    
    