from typing import TypedDict

class AgentState(TypedDict):
    #messages: List[Dict[str, str]] = [] -> sevres as good practice for other apps that exchange messages - react agents or ReWOO
    company_name: str = ''
    found: bool = False
    retrieved_text: str = ''
    scraped_text: str = ''
    board_members: str = ''