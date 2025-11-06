"""
Windows Voice Assistant
========================

A complete voice assistant for Windows that runs locally using open-source tools.

INSTALLATION INSTRUCTIONS:
--------------------------
1. Install Python 3.10 or higher from https://www.python.org/

2. Install required dependencies using pip:
   pip install SpeechRecognition pyttsx3 requests pyaudio
   
   Note: If pyaudio installation fails on Windows, try installing it directly:
   pip install pyaudio
   
   If that still fails, you may need to install Visual C++ Build Tools or use a pre-built wheel.

3. Install Ollama from https://ollama.ai/
   - Download and install Ollama for Windows
   - Run Ollama in the background (it starts automatically after installation)
   - Pull a model (e.g., "ollama pull mistral" or "ollama pull llama3")

4. Run this script:
   python voice_assistant.py

USAGE:
------
- The assistant will start listening for your voice commands
- Say "exit", "quit", "stop", "exit assistant", or "close assistant" to end the session
- For system commands, use phrases like:
  * "Open [app name]" - Opens applications (e.g., "Open Chrome", "Open Notepad")
  * "Close [app name]" - Closes applications (e.g., "Close Chrome", "Close Settings")
  * "Open [folder name]" - Opens folders (e.g., "Open Downloads", "Open Documents folder")
  * "Delete file [path]" - Deletes files with confirmation
- For general questions, just ask naturally - the assistant will use Ollama to respond

EXAMPLE VOICE COMMANDS:
-----------------------
- "Open Chrome" or "Open Google Chrome"
- "Open Notepad"
- "Open Calculator" or "Open Calc"
- "Open Settings"
- "Close Settings" or "Close Chrome"
- "Open File Explorer"
- "Open Downloads"
- "Open Documents folder"
- "Open Pictures"
- "Open Visual Studio Code" or "Open Code"
- "Delete file C:\\Users\\YourName\\Desktop\\test.txt"
- "What is Python?"
- "Tell me a joke"
- "What's the weather like?"
- "Exit"

CONFIGURATION:
--------------
- Default Ollama model: "mistral" (change OLLAMA_MODEL variable to use a different model)
- Ollama API URL: http://localhost:11434 (default Ollama port)
- Speech recognition uses Google Speech Recognition API (requires internet connection)
- TTS uses pyttsx3 with Windows voices
"""

import speech_recognition as sr
import pyttsx3
import requests
import os
import sys
import time
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"  # Change to "llama3" or other model if needed
CONVERSATION_CONTEXT = []  # Store conversation history for context
MAX_CONTEXT_LENGTH = 10  # Maximum number of previous exchanges to keep


class VoiceAssistant:
    """Main voice assistant class that handles speech recognition, TTS, and LLM integration."""
    
    def __init__(self):
        """Initialize the voice assistant with speech recognition and TTS engines."""
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.tts_engine = None
        self.running = True
        
        # Initialize microphone
        try:
            self.microphone = sr.Microphone()
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Microphone initialized successfully.")
        except Exception as e:
            print(f"Error initializing microphone: {e}")
            print("Please ensure your microphone is connected and working.")
            sys.exit(1)
        
        # Initialize TTS engine
        try:
            self.tts_engine = pyttsx3.init()
            # Configure TTS settings
            self.tts_engine.setProperty('rate', 150)  # Speech rate (words per minute)
            self.tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
            # Try to set a Windows voice (if available)
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Prefer a female voice if available, otherwise use default
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            print("Text-to-speech engine initialized successfully.")
        except Exception as e:
            print(f"Error initializing TTS engine: {e}")
            sys.exit(1)
    
    def speak(self, text: str) -> None:
        """
        Convert text to speech and speak it.
        
        Args:
            text: The text to speak
        """
        try:
            print(f"Assistant: {text}")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
    
    def listen(self) -> Optional[str]:
        """
        Listen for voice input and convert to text.
        
        Returns:
            Recognized text or None if recognition failed
        """
        try:
            with self.microphone as source:
                print("Listening... (speak now)")
                # Listen with timeout and phrase time limit
                audio = self.recognizer.listen(
                    source, 
                    timeout=5, 
                    phrase_time_limit=10
                )
            
            print("Processing speech...")
            # Use Google Speech Recognition (works well on Windows, requires internet)
            # This is the most reliable option for Windows
            try:
                text = self.recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                print("Could not understand audio. Please try speaking more clearly.")
                return None
            except sr.RequestError as e:
                print(f"Could not request results from speech recognition service: {e}")
                print("Please check your internet connection.")
                return None
            
            print(f"You said: {text}")
            return text.lower()
            
        except sr.WaitTimeoutError:
            print("Listening timeout - no speech detected.")
            return None
        except Exception as e:
            print(f"Error listening: {e}")
            return None
    
    def query_ollama(self, prompt: str) -> str:
        """
        Query Ollama LLM with the given prompt.
        
        Args:
            prompt: The user's question or prompt
            
        Returns:
            LLM response text
        """
        try:
            # Build context from conversation history
            context_prompt = prompt
            if CONVERSATION_CONTEXT:
                context = "\n".join([
                    f"User: {ctx['user']}\nAssistant: {ctx['assistant']}"
                    for ctx in CONVERSATION_CONTEXT[-MAX_CONTEXT_LENGTH:]
                ])
                context_prompt = f"{context}\nUser: {prompt}\nAssistant:"
            
            # Prepare request payload
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": context_prompt,
                "stream": False
            }
            
            # Make API request
            response = requests.post(
                OLLAMA_API_URL,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', 'I apologize, but I could not generate a response.')
                
                # Update conversation context
                CONVERSATION_CONTEXT.append({
                    'user': prompt,
                    'assistant': answer
                })
                # Keep context manageable
                if len(CONVERSATION_CONTEXT) > MAX_CONTEXT_LENGTH:
                    CONVERSATION_CONTEXT.pop(0)
                
                return answer
            else:
                return f"Error: Ollama API returned status code {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return ("I cannot connect to Ollama. Please ensure Ollama is running "
                   "and accessible at http://localhost:11434")
        except requests.exceptions.Timeout:
            return "The request to Ollama timed out. Please try again."
        except Exception as e:
            return f"An error occurred while querying Ollama: {str(e)}"
    
    def parse_command(self, text: str) -> Tuple[str, Optional[str]]:
        """
        Parse voice command to determine if it's a system command or general query.
        
        Args:
            text: The recognized voice command
            
        Returns:
            Tuple of (command_type, command_data)
            command_type: 'open_app', 'close_app', 'open_folder', 'delete_file', 'exit', or 'general'
            command_data: Additional data for the command (app name, folder name, file path, etc.)
        """
        text = text.strip().lower()
        
        # Close app commands - check BEFORE exit commands
        if text.startswith('close'):
            parts = text.split('close', 1)
            if len(parts) > 1:
                target = parts[1].strip()
                # Remove common words
                target = target.replace('the', '').replace('app', '').replace('application', '').strip()
                
                # Check if it's a known app name
                app_names = [
                    'chrome', 'google chrome', 'firefox', 'edge', 'microsoft edge', 'browser',
                    'notepad', 'notepad++', 'code', 'visual studio code', 'vs code',
                    'calculator', 'calc', 'paint', 'mspaint', 'word', 'excel', 'powerpoint',
                    'spotify', 'discord', 'steam', 'vlc', 'media player',
                    'settings', 'control panel', 'task manager', 'cmd', 'command prompt',
                    'powershell', 'explorer', 'file explorer', 'windows explorer'
                ]
                
                # Check if target matches any app name
                for app in app_names:
                    if app in target or target in app:
                        return ('close_app', app if app in target else target)
        
        # Exit commands - only match standalone exit words or explicit exit phrases
        exit_phrases = ['exit', 'quit', 'stop', 'exit assistant', 'close assistant', 
                        'quit assistant', 'stop assistant', 'shut down', 'shutdown']
        # Check if text is exactly an exit phrase or starts with exit phrase followed by nothing/assistant
        if text in exit_phrases or text.startswith('exit assistant') or text.startswith('close assistant') or text.startswith('quit assistant'):
            return ('exit', None)
        
        # Open commands - check for apps first, then folders
        if text.startswith('open'):
            parts = text.split('open', 1)
            if len(parts) > 1:
                target = parts[1].strip()
                # Remove common words
                target = target.replace('folder', '').replace('the', '').replace('app', '').replace('application', '').strip()
                
                # Check if it's a known app name
                app_names = [
                    'chrome', 'google chrome', 'firefox', 'edge', 'microsoft edge', 'browser',
                    'notepad', 'notepad++', 'code', 'visual studio code', 'vs code',
                    'calculator', 'calc', 'paint', 'mspaint', 'word', 'excel', 'powerpoint',
                    'spotify', 'discord', 'steam', 'vlc', 'media player',
                    'settings', 'control panel', 'task manager', 'cmd', 'command prompt',
                    'powershell', 'explorer', 'file explorer', 'windows explorer'
                ]
                
                # Check if target matches any app name (or contains it)
                for app in app_names:
                    if app in target or target in app:
                        return ('open_app', app if app in target else target)
                
                # If not an app, treat as folder
                return ('open_folder', target)
        
        # Delete file commands
        if text.startswith('delete file') or text.startswith('delete'):
            parts = text.split('delete', 1)
            if len(parts) > 1:
                file_path = parts[1].strip()
                # Remove 'file' if present
                file_path = file_path.replace('file', '').strip()
                return ('delete_file', file_path)
        
        # Default to general query
        return ('general', text)
    
    def open_folder(self, folder_name: str) -> str:
        """
        Open a Windows folder by name.
        
        Args:
            folder_name: Name of the folder (Downloads, Documents, etc.)
            
        Returns:
            Status message
        """
        try:
            # Map common folder names to Windows paths
            folder_map = {
                'downloads': str(Path.home() / 'Downloads'),
                'documents': str(Path.home() / 'Documents'),
                'pictures': str(Path.home() / 'Pictures'),
                'music': str(Path.home() / 'Music'),
                'videos': str(Path.home() / 'Videos'),
                'desktop': str(Path.home() / 'Desktop'),
                'home': str(Path.home()),
            }
            
            folder_name_lower = folder_name.lower()
            
            # Check if it's a mapped folder
            if folder_name_lower in folder_map:
                folder_path = folder_map[folder_name_lower]
            else:
                # Try to use it as a direct path or relative to home
                if os.path.isabs(folder_name):
                    folder_path = folder_name
                else:
                    # Try in user home directory
                    folder_path = str(Path.home() / folder_name)
            
            # Check if folder exists
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                os.startfile(folder_path)
                return f"Opened {folder_name} folder."
            else:
                return f"Sorry, I couldn't find the folder '{folder_name}'."
                
        except Exception as e:
            return f"Error opening folder: {str(e)}"
    
    def open_app(self, app_name: str) -> str:
        """
        Open a Windows application by name.
        
        Args:
            app_name: Name of the application to open
            
        Returns:
            Status message
        """
        try:
            app_name_lower = app_name.lower().strip()
            
            # Map app names to their executable commands or paths
            app_map = {
                # Browsers
                'chrome': 'chrome',
                'google chrome': 'chrome',
                'firefox': 'firefox',
                'edge': 'msedge',
                'microsoft edge': 'msedge',
                'browser': 'msedge',  # Default to Edge
                
                # Text Editors
                'notepad': 'notepad',
                'notepad++': 'notepad++',
                'code': 'code',
                'visual studio code': 'code',
                'vs code': 'code',
                
                # System Tools
                'calculator': 'calc',
                'calc': 'calc',
                'paint': 'mspaint',
                'mspaint': 'mspaint',
                'settings': 'ms-settings:',
                'control panel': 'control',
                'task manager': 'taskmgr',
                'cmd': 'cmd',
                'command prompt': 'cmd',
                'powershell': 'powershell',
                'explorer': 'explorer',
                'file explorer': 'explorer',
                'windows explorer': 'explorer',
                
                # Office (if installed)
                'word': 'winword',
                'excel': 'excel',
                'powerpoint': 'powerpnt',
            }
            
            # Get the command to execute
            if app_name_lower in app_map:
                command = app_map[app_name_lower]
            else:
                # Try using the app name directly (Windows will search PATH)
                command = app_name
            
            # Special handling for some apps
            if command == 'ms-settings:':
                # Windows Settings
                os.startfile('ms-settings:')
                return "Opened Settings."
            elif command == 'explorer':
                # File Explorer
                subprocess.Popen(['explorer'])
                return "Opened File Explorer."
            elif command == 'control':
                # Control Panel
                subprocess.Popen(['control'])
                return "Opened Control Panel."
            elif command == 'taskmgr':
                # Task Manager
                subprocess.Popen(['taskmgr'])
                return "Opened Task Manager."
            elif command in ['cmd', 'powershell']:
                # Command line tools
                subprocess.Popen([command])
                return f"Opened {command}."
            else:
                # Try to find the executable in common locations
                # First, try if it's in PATH
                exe_path = shutil.which(command)
                
                if exe_path:
                    subprocess.Popen([exe_path])
                    return f"Opened {app_name}."
                
                # Try common installation paths
                common_paths = [
                    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
                    "C:\\Program Files\\Mozilla Firefox\\firefox.exe",
                    "C:\\Program Files (x86)\\Mozilla Firefox\\firefox.exe",
                    "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
                    "C:\\Program Files\\Notepad++\\notepad++.exe",
                    "C:\\Program Files\\Microsoft Office\\root\\Office16\\WINWORD.EXE",
                    "C:\\Program Files\\Microsoft Office\\root\\Office16\\EXCEL.EXE",
                    "C:\\Program Files\\Microsoft Office\\root\\Office16\\POWERPNT.EXE",
                    f"{os.environ.get('LOCALAPPDATA', '')}\\Programs\\Microsoft VS Code\\Code.exe",
                ]
                
                # Try to find the app in common paths
                for path in common_paths:
                    if os.path.exists(path):
                        # Check if this path matches our app
                        path_lower = path.lower()
                        if app_name_lower in path_lower or command in path_lower:
                            subprocess.Popen([path])
                            return f"Opened {app_name}."
                
                # Last resort: try using Windows 'start' command
                try:
                    subprocess.Popen(['start', command], shell=True)
                    return f"Opened {app_name}."
                except Exception:
                    # Final fallback: try os.startfile with the command
                    try:
                        os.startfile(command)
                        return f"Opened {app_name}."
                    except Exception:
                        return f"Sorry, I couldn't find or open '{app_name}'. Please make sure it's installed."
                        
        except Exception as e:
            return f"Error opening application: {str(e)}"
    
    def close_app(self, app_name: str) -> str:
        """
        Close a Windows application by name.
        
        Args:
            app_name: Name of the application to close
            
        Returns:
            Status message
        """
        try:
            app_name_lower = app_name.lower().strip()
            
            # Map app names to their process names
            process_map = {
                'chrome': 'chrome.exe',
                'google chrome': 'chrome.exe',
                'firefox': 'firefox.exe',
                'edge': 'msedge.exe',
                'microsoft edge': 'msedge.exe',
                'notepad': 'notepad.exe',
                'notepad++': 'notepad++.exe',
                'code': 'Code.exe',
                'visual studio code': 'Code.exe',
                'vs code': 'Code.exe',
                'calculator': 'Calculator.exe',
                'calc': 'Calculator.exe',
                'paint': 'mspaint.exe',
                'mspaint': 'mspaint.exe',
                'word': 'WINWORD.EXE',
                'excel': 'EXCEL.EXE',
                'powerpoint': 'POWERPNT.EXE',
                'spotify': 'Spotify.exe',
                'discord': 'Discord.exe',
                'steam': 'steam.exe',
                'vlc': 'vlc.exe',
                'settings': 'SystemSettings.exe',
                'cmd': 'cmd.exe',
                'command prompt': 'cmd.exe',
                'powershell': 'powershell.exe',
            }
            
            # Get process name
            if app_name_lower in process_map:
                process_name = process_map[app_name_lower]
            else:
                # Try appending .exe
                process_name = app_name + '.exe'
            
            # Try using psutil first (more reliable)
            try:
                import psutil
                closed_count = 0
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        if proc.info['name'] and proc.info['name'].lower() == process_name.lower():
                            proc.terminate()
                            closed_count += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        pass
                
                if closed_count > 0:
                    return f"Closed {app_name}."
                else:
                    return f"{app_name} is not currently running."
            except ImportError:
                # Fallback to taskkill if psutil is not available
                result = subprocess.run(['taskkill', '/F', '/IM', process_name], 
                                      capture_output=True, text=True, check=False)
                if result.returncode == 0 or 'successfully' in result.stdout.lower():
                    return f"Closed {app_name}."
                elif 'not found' in result.stderr.lower() or 'could not find' in result.stderr.lower():
                    return f"{app_name} is not currently running."
                else:
                    return f"Could not close {app_name}. It may not be running or I don't have permission."
                
        except Exception as e:
            return f"Error closing {app_name}: {str(e)}"
    
    def delete_file(self, file_path: str) -> str:
        """
        Delete a file with confirmation.
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            Status message
        """
        try:
            # Clean up the path
            file_path = file_path.strip()
            
            # Try to resolve the path
            if not os.path.isabs(file_path):
                # Try relative to current directory or home
                if os.path.exists(file_path):
                    resolved_path = os.path.abspath(file_path)
                else:
                    resolved_path = str(Path.home() / file_path)
            else:
                resolved_path = file_path
            
            # Check if file exists
            if not os.path.exists(resolved_path):
                return f"Sorry, I couldn't find the file '{file_path}'."
            
            if os.path.isdir(resolved_path):
                return f"'{file_path}' is a folder, not a file. I can only delete files."
            
            # Ask for confirmation
            self.speak(f"Are you sure you want to delete {os.path.basename(resolved_path)}? Say yes to confirm.")
            confirmation = self.listen()
            
            if confirmation and 'yes' in confirmation.lower():
                os.remove(resolved_path)
                return f"Deleted {os.path.basename(resolved_path)} successfully."
            else:
                return "Deletion cancelled."
                
        except PermissionError:
            return f"Permission denied. I cannot delete '{file_path}'."
        except Exception as e:
            return f"Error deleting file: {str(e)}"
    
    def process_command(self, text: str) -> None:
        """
        Process a voice command and execute the appropriate action.
        
        Args:
            text: The recognized voice command
        """
        if not text:
            return
        
        command_type, command_data = self.parse_command(text)
        
        if command_type == 'exit':
            self.speak("Goodbye! Have a great day.")
            self.running = False
            
        elif command_type == 'open_app':
            response = self.open_app(command_data)
            self.speak(response)
            
        elif command_type == 'close_app':
            response = self.close_app(command_data)
            self.speak(response)
            
        elif command_type == 'open_folder':
            response = self.open_folder(command_data)
            self.speak(response)
            
        elif command_type == 'delete_file':
            response = self.delete_file(command_data)
            self.speak(response)
            
        elif command_type == 'general':
            # Route to Ollama for general conversation
            self.speak("Let me think about that...")
            response = self.query_ollama(text)
            self.speak(response)
    
    def run(self) -> None:
        """Main execution loop for the voice assistant."""
        self.speak("Hello! I'm your voice assistant. How can I help you today?")
        
        while self.running:
            try:
                # Listen for voice input
                text = self.listen()
                
                if text:
                    # Process the command
                    self.process_command(text)
                else:
                    # If no text recognized, wait a bit before listening again
                    time.sleep(0.5)
                    
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                self.speak("Goodbye!")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                self.speak("I encountered an error. Please try again.")


def main():
    """Main entry point for the voice assistant."""
    print("=" * 60)
    print("Windows Voice Assistant")
    print("=" * 60)
    print("\nMake sure Ollama is running before starting the assistant.")
    print("Press Ctrl+C to exit at any time.\n")
    
    try:
        assistant = VoiceAssistant()
        assistant.run()
    except Exception as e:
        print(f"Failed to start voice assistant: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

