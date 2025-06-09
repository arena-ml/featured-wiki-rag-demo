from ollama import chat
import ollama
# import openlit

# openlit.init(collect_gpu_stats=True,capture_message_content=False)

genOpts = {"num_predict":100,"num_ctx":3200}

prompt = f"""
<|user|>
why is sky blue ?
<|end|>

<|assistant|>
"""
result = ollama.generate(model='phi3.5:3.8b-mini-instruct-q8_0', prompt=prompt,options=genOpts)
print(result['response'])

tkip="\"Good American Family\" is an American drama limited series that premiered on Hulu on March 19, 2025, inspired by true events surrounding the adoption of Natalia Grace, a 7-year-old girl with dwarfism. The series, created by Katie Robbins and executive produced by Ellen Pompeo, Mark Duplass, Sarah Sutherland, Mike Epps, Dan Spilo, Niles Kirchner, and Andrew Stearn, explores the complex narrative of Natalia's life through multiple perspectives, focusing on her adoptive parents, Kristine (played by Ellen Pompeo) and Michael Barnett (played by Mark Duplass), and the detective investigating them, Brandon Drysdale (Dulé Hill).\n\nThe series received critical acclaim for its portrayal of the conflicting narratives and multifaceted characters, with particular praise directed at the performances of Imogen Faith Reid as Natalia Grace and Ellen Pompeo as Kristine Barnett. Critics recognized the show's exploration of themes such as morality, human nature, guilt, justice, and redemption, despite some criticism of its sensationalism and tone. Good American Family ranked highly in viewership, being the sixth most-streamed series and the number one on Hulu's \"Top 15 Today\" list for the week ending March 26, 2025.\n\nThe series also received a nomination for Location Team of the Year - Episodic Television - One Hour at the 2024 California on Location Awards.\n\nRecent changes in the article include a rollback edit to address unexplained content removal and a revert to the last revision for clarity. The production details were also updated to reflect the accurate casting and the principal photography location in Santa Clarita, California. The reception section was revised to include critical perspectives on the show's narrative style and thematic exploration, as well as its impact on viewers and the real-life case it portrays.\n\nKey aspects of the series include its unique narrative structure, strong character performances, exploration of moral themes, and critical acclaim, despite some critiques of its tone and sensationalism. The series' international availability on Disney+ and its viewership success underscore its relevance and impact in the realm of streaming television dramas."
tokens = ollama.embed(model="phi3.5:3.8b-mini-instruct-q8_0",input=tkip)
print(tokens.prompt_eval_count)

