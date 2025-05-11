import subprocess
from langchain.tools import StructuredTool, BaseTool

class GenericTools:
    @staticmethod
    def create_open_app_tool() -> BaseTool:
        def open_app(app_name: str) -> str:
            try:
                subprocess.Popen(["open", "-a", app_name])
                return ""  # Return empty string for success (to suppress response)
            except Exception as e:
                return f"Failed to open {app_name}: {str(e)}"

        return StructuredTool.from_function(
            func=open_app,
            name="open_app",
            description="Opens an application on the device by name using subprocess."
        )
