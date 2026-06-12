uv — what it is and how to use it                                                          
                                                                                            
 uv is a single Rust binary that replaces a bunch of Python tooling. Think of it as cargo   
 for Python, written by the same people who made ruff. It handles:                          
                                                                                            
 - Installing Python versions (no more pyenv, python.org installers, or Homebrew Python     
   juggling)                                                                                
 - Creating virtual environments                                                            
 - Installing packages                                                                      
 - Resolving and locking dependencies                                                       
 - Running scripts with inline dependencies                                                 
 - Building/publishing packages (we won't use this)                                         
                                                                                            
 Why it's nice for our project:                                                             
 - One tool, one config file (pyproject.toml), no requirements.txt + venv + pip-compile     
   dance                                                                                    
 - uv sync gives you a deterministic environment from a lockfile                            
 - It's fast — installs that take 30s with pip take 2s with uv                              
 - It can install Python 3.12 even though your system is on 3.14, no pyenv required         
                                                                                            
 ### Installation                                                                           
                                                                                            
 ```bash                                                                                    
   # macOS / Linux, one command:                                                            
   curl -LsSf https://astral.sh/uv/install.sh | sh                                          
                                                                                            
   # Or via Homebrew:                                                                       
   brew install uv                                                                          
 ```                                                                                        
                                                                                            
 That puts uv in ~/.local/bin (the curl way) or /opt/homebrew/bin (brew). Add ~/.local/bin  
 to your PATH if the installer says it's not found.                                         
                                                                                            
 Verify:                                                                                    
                                                                                            
 ```bash                                                                                    
   uv --version                                                                             
 ```                                                                                        
                                                                                            
 ### The commands you'll actually use                                                       
                                                                                            
 ```bash                                                                                    
   # Install a specific Python version (one-time per machine)                               
   uv python install 3.12                                                                   
                                                                                            
   # Pin a Python version for this project (creates .python-version file)                   
   uv python pin 3.12                                                                       
                                                                                            
   # Create a virtual env and activate it the old way, OR:                                  
   # (uv auto-manages the venv in .venv/ for you)                                           
                                                                                            
   # Install all dependencies from pyproject.toml                                           
   uv sync                                                                                  
                                                                                            
   # Add a new dependency (updates pyproject.toml + uv.lock automatically)                  
   uv add typer                                                                             
   uv add --dev pytest                                                                      
                                                                                            
   # Remove a dependency                                                                    
   uv remove some-package                                                                   
                                                                                            
   # Run a command inside the project's environment                                         
   uv run ytx "https://youtu.be/..."                                                        
   uv run pytest                                                                            
   uv run python -c "print('hi')"                                                           
                                                                                            
   # Activate the venv manually if you want a shell session in it                           
   source .venv/bin/activate                                                                
 ```                                                                                        
                                                                                            
 ### How this maps to our project                                                           
                                                                                            
 Instead of this:                                                                           
                                                                                            
 ```bash                                                                                    
   python3.12 -m venv .venv                                                                 
   source .venv/bin/activate                                                                
   pip install -r requirements.txt                                                          
   python -m youtubetranscriber.cli ...                                                     
 ```                                                                                        
                                                                                            
 You do this:                                                                               
                                                                                            
 ```bash                                                                                    
   uv sync          # first time, and whenever deps change                                  
   uv run ytx ...   # runs in the project env automatically                                 
 ```                                                                                        
                                                                                            
 ### One gotcha worth knowing                                                               
                                                                                            
 uv add updates both pyproject.toml (the loose pins) and uv.lock (the exact resolved        
 versions for reproducibility). Always commit both to git. If you see "works on my machine" 
 issues, the lockfile is the answer.                                                        
                                                                                            
 ### That's it                                                                              
                                                                                            
 There's a lot more uv can do (workspaces, tool management, etc.) but you don't need any of 
 it. The five commands above cover 95% of what we'll do.                                    
