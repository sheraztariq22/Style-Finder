"""
Service for interacting with Google Gemini Vision models.
"""

import logging
import google.generativeai as genai
from PIL import Image
import io
import base64

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiVisionService:
    """
    Provides methods to interact with Google Gemini Vision models.
    """
    
    def __init__(self, api_key, model_name="gemini-2.0-flash-exp", 
                 temperature=0.2, top_p=0.6, max_tokens=2000):
        """
        Initialize the service with the specified model and parameters.
        
        Args:
            api_key (str): Google API key for authentication
            model_name (str): Name of the Gemini model to use
            temperature (float): Controls randomness in generation
            top_p (float): Nucleus sampling parameter
            max_tokens (int): Maximum tokens in the response
        """
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Set up generation configuration
        self.generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
        }
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.generation_config
        )
        
        logger.info(f"Initialized Gemini model: {model_name}")
    
    def generate_response(self, encoded_image, prompt):
        """
        Generate a response from the model based on an image and prompt.
        
        Args:
            encoded_image (str): Base64-encoded image string
            prompt (str): Text prompt to guide the model's response
            
        Returns:
            str: Model's response
        """
        try:
            logger.info("Sending request to Gemini with prompt length: %d", len(prompt))
            
            # Decode base64 image
            image_data = base64.b64decode(encoded_image)
            image = Image.open(io.BytesIO(image_data))
            
            # Create the content parts
            content = [prompt, image]
            
            # Generate response
            response = self.model.generate_content(content)
            
            # Extract the text from response
            content_text = response.text
            
            logger.info("Received response with length: %d", len(content_text))
            
            # Check if response appears to be truncated
            if len(content_text) >= 7900:
                logger.warning("Response may be truncated (length: %d)", len(content_text))
            
            return content_text
            
        except Exception as e:
            logger.error("Error generating response: %s", str(e))
            return f"Error generating response: {e}"
    
    def generate_fashion_response(self, user_image_base64, matched_row, all_items, 
                                 similarity_score, threshold=0.8):
        """
        Generate a fashion-specific response using role-based prompts.
        
        Args:
            user_image_base64: Base64-encoded user-uploaded image
            matched_row: The closest match row from the dataset
            all_items: DataFrame with all items related to the matched image
            similarity_score: Similarity score between user and matched images
            threshold: Minimum similarity for considering an exact match
            
        Returns:
            str: Detailed fashion response
        """
        # Generate a simpler list of items with prices and links
        items_list = []
        for _, row in all_items.iterrows():
            item_str = f"{row['Item Name']} (${row['Price']}): {row['Link']}"
            items_list.append(item_str)
        
        # Join items with clear separators
        items_description = "\n".join([f"- {item}" for item in items_list])

        if similarity_score >= threshold:
            # Simplified prompt focused on professional fashion analysis
            assistant_prompt = (
                f"You're conducting a professional retail catalog analysis. "
                f"This image shows standard clothing items available in department stores. "
                f"Focus exclusively on professional fashion analysis for a clothing retailer. "
                f"ITEM DETAILS (always include this section in your response):\n{items_description}\n\n"
                "Please:\n"
                "1. Identify and describe the clothing items objectively (colors, patterns, materials)\n"
                "2. Categorize the overall style (business, casual, etc.)\n"
                "3. Include the ITEM DETAILS section at the end\n\n"
                "This is for a professional retail catalog. Use formal, clinical language."
            )
        else:
            # Similar approach for non-exact matches
            assistant_prompt = (
                f"You're conducting a professional retail catalog analysis. "
                f"This image shows standard clothing items available in department stores. "
                f"Focus exclusively on professional fashion analysis for a clothing retailer. "
                f"SIMILAR ITEMS (always include this section in your response):\n{items_description}\n\n"
                "Please:\n"
                "1. Note these are similar but not exact items\n"
                "2. Identify clothing elements objectively (colors, patterns, materials)\n" 
                "3. Include the SIMILAR ITEMS section at the end\n\n"
                "This is for a professional retail catalog. Use formal, clinical language."
            )
        
        # Send the prompt to the model
        response = self.generate_response(user_image_base64, assistant_prompt)
        
        # Check if response is incomplete
        if len(response) < 100:
            logger.info("Response appears incomplete, creating basic response")
            # Create a basic response with the item details
            section_header = "ITEM DETAILS:" if similarity_score >= threshold else "SIMILAR ITEMS:"
            response = f"# Fashion Analysis\n\nThis outfit features a collection of carefully coordinated pieces.\n\n{section_header}\n{items_description}"
        
        # Ensure the items list is included - this is crucial
        elif "ITEM DETAILS:" not in response and "SIMILAR ITEMS:" not in response:
            logger.info("Item details section missing from response")
            # Append to existing response
            section_header = "ITEM DETAILS:" if similarity_score >= threshold else "SIMILAR ITEMS:"
            response += f"\n\n{section_header}\n{items_description}"
        
        return response