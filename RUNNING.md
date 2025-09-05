# Running the Autonomous Coding Agent

This system runs on **Linux, macOS, and Windows**.

---

## 1. Install Docker

### Linux (Ubuntu/Debian)
```bash
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
docker --version
```

### macOS
- Install Docker Desktop from [Docker Desktop](https://www.docker.com/products/docker-desktop)
- Start Docker Desktop before running the agent.

### Windows
- Install Docker Desktop from [Docker Desktop](https://www.docker.com/products/docker-desktop)
- Enable **WSL2 Backend**.
- Run all commands in **PowerShell** or **Anaconda Prompt**.

---

## 2. Install Conda

### Linux & macOS
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### Windows
- Download [Miniconda](https://docs.anaconda.com/free/miniconda/)
- Install and run commands in **Anaconda Prompt**.

---

## 3. Create Environment
```bash
conda create -n agent-env python=3.10 -y
conda activate agent-env
pip install -r requirements.txt
```

---

## 4. Build Docker Sandbox
```bash
docker build -t agent-sandbox:latest ./sandbox
```

---

## 5. Run the Agent
```bash
streamlit run app.py
```
Access UI at [http://localhost:8501](http://localhost:8501)

---

## ⚠️ OS Notes
- **Windows**:
  - Use **Anaconda Prompt** or **PowerShell**.
  - Paths auto-normalized (C:/Users/...).

- **macOS/Linux**:
  - Ensure user is in `docker` group to run without `sudo`:
    ```bash
    sudo usermod -aG docker $USER
    ```
