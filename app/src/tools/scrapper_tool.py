import time
import re # regex expressions

from selenium import webdriver
from selenium.webdriver.firefox.options import Options

from googlesearch import search

from bs4 import BeautifulSoup

from src.tools.tool import Tool


class ScrapperTool(Tool):

    def __init__(self):
        super().__init__()

    def invoke_tool(self, company_name: str):
        return  self._scrape_top_result(company_name)

    def _scrape_top_result(self, company_name):
        """ Perform Google Search given the company name as a filter and scrape first result """
        search_query = f"board members of {company_name}"
        search_results = list(search(search_query,  region="us", num_results=5))
        if not search_results:
            return None
        url = search_results[0]
        # selenium helps to get all items loaded trough javascipt in the site
        try:
            options = Options()
            options.add_argument("--headless=new")
            driver = webdriver.Firefox(options=options)
            driver.get(url)
            html = driver.page_source
            time.sleep(2)
        finally:
            driver.quit()

        return self._extract_clean_text_from_html(html)
    

    def _extract_clean_text_from_html(self, html_content: str) -> str:
        
        soup = BeautifulSoup(html_content, "lxml")

        # Completely remove script, style, and irrelevant metadata tags
        for tag in soup(["script", "style", "noscript", "meta", "link", "iframe", "svg"]):
            tag.decompose()

        # Remove comments
        for comment in soup.find_all(string=lambda s: isinstance(s, (type(soup.Comment)))):
            comment.extract()

        raw_text = soup.get_text(separator=' ')
        cleaned_text = re.sub(r'\s+', ' ', raw_text)  # normalize whitespace
        cleaned_text = re.sub(r'[{<\[].*?[>\]}]', '', cleaned_text)  # remove residual brackets, e.g., [1], <tag>
        
        return cleaned_text.strip()