import numpy as np

class ImageCompressor:
    """
      This class is responsible to
          1. Learn the codebook given the training images
          2. Compress an input image using the learnt codebook
    """
    def __init__(self):
        """
        Feel free to add any number of parameters here.
        But be sure to set default values. Those will be used on the evaluation server
        """
        
        # Here you can set some parameters of your algorithm, e.g.
        #self.dtype = np.float16

        self.k = 18
        self.codebook = np.empty((4, 9216, self.k), dtype=np.float32)
        


    def get_codebook(self):
        """ Codebook contains all information needed for compression/reconstruction """
        
        
        return self.codebook
    
    def preprocess(self, train_images):
      processed = []
      for image in train_images:

        threshold = 100  
        # Calculate the Euclidean distance to white (255, 255, 255) for each pixel
        distances_to_white = np.linalg.norm(image - [255, 255, 255], axis=-1)
        # Create a mask for pixels near white based on the threshold
        near_white_mask = distances_to_white < threshold
        # Set pixels near white to pure white (255, 255, 255)
        new_image = image.copy()
        new_image[near_white_mask] = [255, 255, 255]

        processed.append(new_image)
      return processed
    


    def train(self, train_images):
        """
        Training phase of your algorithm - e.g. here you can perform PCA on training data
        
        Args:
            train_images  ... A list of NumPy arrays.
                              Each array is an image of shape H x W x C, i.e. 96 x 96 x 3
        """
        
        
        images = np.float32(train_images)
        images = self.preprocess(images)

        #mean
        mean = np.mean(images, axis=0)
        images = images - mean
        
        mean2save = mean
        self.codebook[0, :9217, :3] = mean2save.reshape(9216, 3)

        for i in range(3): #for each color channel

            images_1channel = images[:,:,:,i]
            #print(images_1channel.shape)

            images_1channel = images_1channel.reshape(100, -1)  #flatten images
            #print(images_1channel.shape)

            #covariance matrix
            cov_matrix = np.cov(images_1channel, rowvar=False)
            #print(cov_matrix.shape)

            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            sorted_indices = np.argsort(eigenvalues)[::-1]
            sorted_eigenvalues = eigenvalues[sorted_indices]
            sorted_eigenvectors = eigenvectors[:,sorted_indices]
            

            pc_amount = self.k
            pc_eigenvectors = sorted_eigenvectors[:,0:pc_amount]
            self.codebook[i + 1, :, :] = pc_eigenvectors
        
    



    def compress(self, test_image):
        """ Given an array of shape H x W x C return compressed code """

        test_image = test_image.astype(np.uint8)
        mean = self.codebook[0, :9217, :3].reshape(96, 96, 3) #codebook mean
        
        test_image = test_image - mean


        red_channel = test_image[:, :, 0]
        green_channel = test_image[:, :, 1]
        blue_channel = test_image[:, :, 2]

        test_image_r = red_channel.reshape(-1, 3)
        test_image_g = green_channel.reshape(-1, 3)
        test_image_b = blue_channel.reshape(-1, 3)

        
        #get codebook eigenvectors
        eigenvectors_red = self.codebook[1]
        eigenvectors_green = self.codebook[2]
        eigenvectors_blue = self.codebook[3]

        reduced_r = np.dot(test_image_r.reshape(1, -1), eigenvectors_red)
        reduced_g = np.dot(test_image_g.reshape(1, -1), eigenvectors_green)
        reduced_b = np.dot(test_image_b.reshape(1, -1), eigenvectors_blue)


        reduced = np.dstack([reduced_r, reduced_g, reduced_b])
        reduced = reduced.astype(np.int16)
        normalized_arr = (reduced - np.min(reduced)) / (np.max(reduced) - np.min(reduced))
        reduced_8 = normalized_arr * 255
        

        reduced_8 = np.append(reduced_8, [np.abs(np.min(reduced) / 30), np.max(reduced) / 30])
        reduced_8 = reduced_8.astype(np.uint8)

        return reduced_8


class ImageReconstructor:
    """ This class is used on the server to reconstruct images """
    def __init__(self, codebook):
        """ The only information this class may receive is the codebook """
        self.codebook = codebook

    def reconstruct(self, test_code):
        """ Given a compressed code of shape K, reconstruct the original image """

        min = test_code[-2] * -30 
        max = test_code[-1] * 30
        
        test_code = test_code[:-2].reshape(1, -1, 3)

        test_code = ((test_code / 255) * (max - min) + min).astype(np.int16)

        reduced_r = test_code[:, :, 0] 
        reduced_g = test_code[:, :, 1]
        reduced_b = test_code[:, :, 2]

        #get codebook eigenvectors
        eigenvectors_red = self.codebook[1] 
        eigenvectors_green = self.codebook[2]
        eigenvectors_blue = self.codebook[3]
        
        mean = self.codebook[0, :9217, :3].reshape(96, 96, 3)
        

        reconstructed_r = np.dot(reduced_r, eigenvectors_red.T) 
        reconstructed_r = reconstructed_r.reshape(96, 96, -1)
        reconstructed_g = np.dot(reduced_g, eigenvectors_green.T) 
        reconstructed_g = reconstructed_g.reshape(96, 96, -1) 
        reconstructed_b = np.dot(reduced_b, eigenvectors_blue.T) 
        reconstructed_b = reconstructed_b.reshape(96, 96, -1) 

        reconstructed = np.dstack([reconstructed_r, reconstructed_g, reconstructed_b]) + mean
        
        reconstructed = reconstructed.astype(np.uint32)
        reconstructed[reconstructed > 1000] = 0
        reconstructed = np.clip(reconstructed, 0, 255) 

        #Near white to white

        threshold = 50  # You can adjust this threshold as needed
        # Calculate the Euclidean distance to white (255, 255, 255) for each pixel
        distances_to_white = np.linalg.norm(reconstructed - [255, 255, 255], axis=-1)
        # Create a mask for pixels near white based on the threshold
        near_white_mask = distances_to_white < threshold
        # Set pixels near white to pure white (255, 255, 255)
        reconstructed[near_white_mask] = [255, 255, 255]



        return reconstructed