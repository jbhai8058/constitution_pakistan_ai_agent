# Setup Guide for Constitution Q&A Agent

## Step 1: Install Dependencies

Open your terminal in the project directory and run:

```bash
pip install -r requirements.txt
```

## Step 2: Create .env File

Create a file named `.env` in the root directory with the following content:

```
OPENAI_API_KEY=your-actual-api-key-here
```

**Important:** 
- Replace `your-actual-api-key-here` with your actual OpenAI API key
- Get your API key from: https://platform.openai.com/api-keys
- Never commit the `.env` file to version control

## Step 3: Process the PDF (First Time Only)

Run the ingestion script to process the Constitution PDF and create the vector store:

```bash
python ingest.py
```

This will:
- Load the PDF from `Ai/constitution.pdf`
- Split it into chunks
- Create embeddings
- Save the vector store to `vector_store/` folder

**Note:** This step only needs to be run once, or whenever you update the PDF.

## Step 4: Start the Chat Interface

Run the main script to start asking questions:

```bash
python main.py
```

Type your questions about the Constitution and press Enter. Type `quit`, `exit`, or `q` to stop.

## Troubleshooting

### "Vector store not found" error
- Make sure you ran `python ingest.py` first

### "API key not found" error
- Check that your `.env` file exists and contains `OPENAI_API_KEY=your-key`
- Make sure there are no extra spaces or quotes around the key

### Import errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- You might need to upgrade pip: `pip install --upgrade pip`

