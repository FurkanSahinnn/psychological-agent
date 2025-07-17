import sys
from rich.console import Console
from rich.markdown import Markdown
from langgraph_pipeline import psychology_agent, AgentState

def run_agent(query: str):
    """
    Runs the psychology agent with the given query and prints the result to the console.
    """
    console = Console()
    
    console.print(f"[bold cyan]🔍 User Question:[/bold cyan] {query}\n")

    initial_state = AgentState(user_query=query, retrieved_documents=None, final_prompt=None, final_response=None)

    events = psychology_agent.stream(initial_state)

    final_response = ""
    for event in events:
        if "final_response" in event.get("generator", {}):
            final_response = event["generator"]["final_response"]
    
    console.print("\n[bold green]✅ Psychology Agent's Answer:[/bold green]")
    
    # Format the response to be more readable by using Markdown 
    markdown_response = Markdown(final_response)
    console.print(markdown_response)

def main():
    """
    Reads a query from the user or from the command line as an argument.
    """
    if len(sys.argv) > 1:
        # Use the query provided as an argument
        query = " ".join(sys.argv[1:])
        run_agent(query)
    else:
        console = Console()
        console.print("[bold magenta]Welcome to the Psychology Agent![/bold magenta]")
        console.print("Enter your question and press 'Enter'. To exit, type 'exit'.")
        
        while True:
            query = console.input("[bold cyan]Your Question: [/bold cyan]")
            if query.lower() in ["exit", "quit", "exit"]:
                break
            if query.strip():
                run_agent(query)
            else:
                console.print("[red]Please enter a question.[/red]")

if __name__ == "__main__":
    try:
        import langchain
        import langgraph
        import qdrant_client
        import openai
        import dotenv
        import rich
    except ImportError:
        print("Required libraries are missing. Please run 'pip install -r requirements.txt'.")
        sys.exit(1)
        
    main()
