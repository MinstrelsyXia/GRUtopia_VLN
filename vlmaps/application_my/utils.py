
class NotFound(Exception):
    """Exception raised when a specific object is not found on the map."""
    def __init__(self, message="Object not found or insufficient data in pc_mask."):
        self.message = message
        super().__init__(self.message)
