🔧 Setup Instructions

 1. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

2. 🔐 Set Up Hugging Face API Token

You have two options:

Option A – Use a `.env` File (Recommended)

Create a `.env` file in your project root:

```bash
touch .env
```

Add your Hugging Face API token to it:

```env
HF_API_TOKEN=your_huggingface_token_here
```

> The app will automatically load this token using `python-dotenv`.

Option B – Set the Environment Variable Directly (Not persistent)

```bash
export HF_API_TOKEN=your_huggingface_token_here
```

> You must run this command **every time you restart your terminal**.

---

3. 🚀 Run the FastAPI Server

```bash
uvicorn main:app --reload
```

Visit your API docs at: [http://localhost:8000/docs](http://localhost:8000/docs)
