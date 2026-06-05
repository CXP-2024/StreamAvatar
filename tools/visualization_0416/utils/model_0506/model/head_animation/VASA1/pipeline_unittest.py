import unittest
import torch
import torchvision
import os
from face_encoder import FaceEncoder
from motion_encoder import MotionEncoder
from flow_estimator import FlowEstimator
from face_generator import Generator


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = 512
        self.latent_dim = 128 # this value is not reported in the MagaPorait paper
        self.use_gt_rotation = True
        self.hopenet_checkpoint_path = "/mnt/weka/real_time_data/model/hopenet_robust_alpha1.pkl"
        self.normalize_output=True
        # self.resize_motion_encoder_input = False

        ## Epp Part
        self.face_encoder = FaceEncoder(self.latent_dim, self.normalize_output).to(self.device)

        ## Emnt part
        self.motion_encoder = MotionEncoder(latent_dim=self.latent_dim, size=self.img_size, normalize_output=self.normalize_output, use_gt_rotation=self.use_gt_rotation, hopenet_checkpoint_path=self.hopenet_checkpoint_path).to(self.device)

        # flow estimator
        self.flow_estimator = FlowEstimator(latent_dim=self.latent_dim).to(self.device)

        # face generator
        self.generator = Generator().to(self.device)

    def test_face_encoder(self):
        """Test if the output shape is correct for a given input shape"""
        batch_size = 2
        channels = 3

        # Create a random input tensor
        input_tensor = torch.randn(batch_size, channels, self.img_size, self.img_size).to(self.device)

        # Get output from the model
        with torch.no_grad():
            [feature_volume,  global_descriptor]= self.face_encoder(input_tensor)
        print('Feature_volume shape:', feature_volume.shape)
        print('Global_descriptor shape:', global_descriptor.shape)

    def test_motion_encoder(self):
        """Test if the output shape is correct for a given input shape"""
        batch_size = 2
        channels = 3

        # Create a random input tensor
        input_tensor = torch.randn(batch_size, channels, self.img_size, self.img_size).to(self.device)

        # Get output from the model
        with torch.no_grad():
            motion_code, rigid_pose = self.motion_encoder(input_tensor)

        print('Motion code shape:',  motion_code.shape)
        for k, v in rigid_pose.items():
            print(f'Rigid pose output: {k}',  v.shape)

    def test_flow_estimator(self):
        """Test if the output shape is correct for a given input shape"""
        batch_size = 2
        src_dict = {}
        src_dict["feature_volume"] = torch.randn(batch_size, 96, 16, 64, 64).to(self.device)
        src_dict["global_descriptor"] = torch.randn(batch_size, self.latent_dim).to(self.device)
        src_dict["expression_code"] = torch.randn(batch_size, self.latent_dim).to(self.device)
        src_dict["rigid_pose"] = {"rotation": torch.randn(batch_size, 3, 3).to(self.device), "translation": torch.randn(batch_size, 3).to(self.device)}

        tgt_dict = {}
        tgt_dict["expression_code"] = torch.randn(batch_size, self.latent_dim).to(self.device)
        tgt_dict["rigid_pose"] = {"rotation": torch.randn(batch_size, 3, 3).to(self.device), "translation": torch.randn(batch_size, 3).to(self.device)}

        canonical_feature_volume, warped_driving_feature_volume = self.flow_estimator(src_dict, tgt_dict)

        print('Canonical feature volume shape:', canonical_feature_volume.shape)
        print('Warped driving feature volume shape:', warped_driving_feature_volume.shape)

    def test_generator(self):
        """
        Test if the output shape is correct for a given input shape
        """
        batch_size = 2
        warped_driving_feature_volume = torch.randn(batch_size, 96, 16, 64, 64).to(self.device)
        predicted_image = self.generator(warped_driving_feature_volume)
        print('Predicted image shape:',  predicted_image.shape)

        assert predicted_image.shape == (batch_size, 3, self.img_size, self.img_size)

if __name__ == '__main__':
    unittest.main()
