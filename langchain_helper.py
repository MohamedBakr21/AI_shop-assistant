from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("API_KEY"),
    model="llama3-70b-8192"
)

# Define JSON output structure
class SerpAPIPromptResponse(BaseModel):
    refined_query: str = Field(description="Refined search query for the product search.")
    additional_info: str = Field(description="Additional adjectives summarized to be added to the search query.")

# JSON output parser
json_parser = JsonOutputParser(pydantic_object=SerpAPIPromptResponse)

# LLM Prompt Template
llama_template = PromptTemplate(
    template="System: {system_prompt}\n{format_prompt}\nHuman: {user_prompt}\nAI:",
    input_variables=["system_prompt", "format_prompt", "user_prompt"]
)

def llm_generate_gl(location: str) -> str | None:
    """
    Use the LLM to determine the ISO 3166-1 alpha-2 country code (gl) from a location string.
    Returns the two-letter country code or None if invalid.
    """
    system_prompt = (
        "You are an expert at identifying ISO 3166-1 alpha-2 country codes from locations. "
        "Output only the two-letter country code (e.g., 'US' for United States) for the provided location, "
        "without any additional text or explanations."
    )
    format_prompt = "Output the ISO 3166-1 alpha-2 country code only."
    user_prompt = f"Location: {location}"

    try:
        # Create the chain: prompt template + llm
        chain = llama_template | llm

        # Invoke LLM; result is an AIMessage object
        response = chain.invoke({
            "system_prompt": system_prompt,
            "format_prompt": format_prompt,
            "user_prompt": user_prompt
        })

        # Extract the text content from AIMessage and strip whitespace
        gl_code = response.content.strip()

        # Validate the result
        if len(gl_code) == 2 and gl_code.isalpha():
            print(f"Generated country code (gl): {gl_code.upper()}")
            return gl_code.upper()
        else:
            print(f"Invalid country code returned: {gl_code}")
            return None

    except Exception as e:
        print(f"Error in llm_generate_gl: {e}")
        return None


def refine_query(user_input):
    system_prompt = "You are a highly skilled shopping assistant. Refine user queries into specific product searches."
    format_prompt = json_parser.get_format_instructions()
    chain = llama_template | llm | json_parser
    response = chain.invoke({
        "system_prompt": system_prompt,
        "format_prompt": format_prompt,
        "user_prompt": user_input
    })
    return response


def llm_generate_summary(comparison_df_llm):
    system_prompt = (
        "You are a highly skilled shopping assistant with expertise in comparing products and summarizing findings."
    )
    format_prompt = (
        "<h3>Best Value Product</h3>: Explain best value.\n"
        "<h3>Highest Rated Option</h3>: Highlight the top-rated product.\n"
        "<h3>Unique Features</h3>: List unique features.\n"
        "<h3>Trade-offs and Comparisons</h3>: Discuss trade-offs.\n"
        "<h3>Conclusion and Suggestion</h3>: Final recommendation.\n"
        "Use HTML with <ul> and <li> where needed."
    )
    user_prompt = f"Here is the product information in JSON format:\n{comparison_df_llm.to_dict(orient='records')}"

    try:
        chain = llama_template | llm
        response = chain.invoke({
            "system_prompt": system_prompt,
            "format_prompt": format_prompt,
            "user_prompt": user_prompt
        })
        print("Response type:", type(response))
        print("Response content preview:", response.content[:500])
        summary = response.content.strip()

        required_sections = [
            "<h3>Best Value Product</h3>",
            "<h3>Highest Rated Option</h3>",
            "<h3>Unique Features</h3>",
            "<h3>Trade-offs and Comparisons</h3>",
            "<h3>Conclusion and Suggestion</h3>",
        ]
        for section in required_sections:
            if section not in summary:
                summary += f"\n{section}\n<p>Data unavailable for this section.</p>"
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "<h3>Summary Unavailable</h3><p>An error occurred.</p>"


def generate_comparison_table(products, featured_results=None, nearby_results=None):
    all_products = products[:5]
    product_summaries = [
        {
            "Name": f"<a href='{p.get('product_link', '#')}'>{p.get('title', 'N/A')}</a>",
            "Price Now": p.get("price", "N/A"),
            "Original Price": p.get("old_price", "N/A"),
            "Special Offer": ", ".join(p.get("extensions", [])) if p.get("extensions") else "No Discount",
            "Rating": f"{p.get('rating', 'N/A')} ⭐",
            "Reviews": f"{p.get('reviews', 'N/A')} Reviews",
            "Store": p.get("source", "N/A"),
            "Delivery": p.get("delivery", "N/A"),
            "Image": f"<img src='{p.get('thumbnail', '#')}' style='height:50px;'/>"
        }
        for p in all_products
    ]
    product_summaries_llm = [
        {
            "Name": p.get("title", "N/A"),
            "Price Now": p.get("price", "N/A"),
            "Original Price": p.get("old_price", "N/A"),
            "Special Offer": ", ".join(p.get("extensions", [])) if p.get("extensions") else "No Discount",
            "Rating": f"{p.get('rating', 'N/A')}",
            "Reviews": f"{p.get('reviews', 'N/A')} Reviews",
            "Store": p.get("source", "N/A"),
            "Delivery": p.get("delivery", "N/A"),
        }
        for p in all_products
    ]

    comparison_df = pd.DataFrame(product_summaries)
    comparison_df_llm = pd.DataFrame(product_summaries_llm)

    if comparison_df.empty:
        return "<h3>No Products Found</h3>", "<h3>Summary Unavailable</h3><p>No product data available.</p>"

    summary = llm_generate_summary(comparison_df_llm)
    comparison_table_html = comparison_df.to_html(index=False, escape=False)

    return comparison_table_html, summary
