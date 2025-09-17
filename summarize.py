from transformers import pipeline

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

long_text = """
Whales, the colossal and enigmatic marine mammals, are among the most fascinating creatures on Earth. These giants of the deep have captured human imagination for centuries, but their true story, from their land-dwelling origins to their precarious present, is a testament to the remarkable power of evolution and the challenges posed by human activity.

Origin and Evolution of Whales 🐋
The story of whales begins on land, roughly 50 million years ago, long after the dinosaurs went extinct. Their ancestors were artiodactyls, a group of four-legged, hoofed mammals that also includes modern-day hippos, cows, and deer. The fossil record, particularly from the regions of India and Pakistan, reveals a gradual transition from land-based to fully aquatic life. One of the earliest known whale ancestors is Pakicetus, a wolf-sized carnivore that lived near the water but was still primarily terrestrial. Over millions of years, its descendants developed a number of adaptations for an aquatic existence, including streamlined bodies, forelimbs that transformed into flippers, and powerful tail flukes for propulsion. The nostrils of these ancient whales also migrated backward on the skull, eventually becoming the blowhole on top of the head. This remarkable evolutionary journey is one of the most compelling examples of natural selection in action.






Species and Classification
Today, the infraorder Cetacea, which includes whales, dolphins, and porpoises, comprises around 90 species. These are broadly divided into two major groups:


Mysticetes (Baleen Whales): These whales lack teeth and instead have baleen plates, large, fibrous filters made of keratin that hang from their upper jaws. They feed by taking in huge gulps of water and filtering out small prey like krill and small fish. This group includes the largest animals on the planet, such as the blue whale, fin whale, humpback whale, and gray whale.



Odontocetes (Toothed Whales): This group has teeth and are active hunters. Their diet consists of larger prey, including fish, squid, and even other marine mammals. They are also known for their sophisticated use of echolocation—producing sound waves and interpreting the returning echoes to navigate and locate prey. The odontocete group includes the sperm whale, orca (killer whale), and beluga whale, as well as all dolphins and porpoises.



Physical and Behavioral Characteristics
Whales are warm-blooded, air-breathing mammals that give birth to live young and nurse them with milk. They possess a number of specialized adaptations for life in the ocean. Their bodies are highly streamlined, with a thick layer of blubber for insulation. Their flukes move vertically to propel them through the water, while their flippers are used for steering. Whales have a powerful and rapid exhalation at the surface, which creates the distinctive "spout" or "blow" that can be used to identify different species.




Whales are also highly intelligent and social creatures. Many species live in complex social groups called pods, which are often led by a matriarchal female. They exhibit a range of complex behaviors, including cooperative hunting, where groups work together to herd and trap prey. Humpback whales, for example, are known for their spectacular "bubble-net feeding," where they create a wall of bubbles to corral fish.



""" # The same long text you provided

# Step 1: Split the long text into smaller chunks
# We'll use a simple character count, a rough proxy for token count (a token is often 1.3-1.5 characters)
max_char_count = 1000  # Approximately 700 tokens
chunks = [long_text[i:i + max_char_count] for i in range(0, len(long_text), max_char_count)]

# Step 2: Summarize each chunk
summaries = []
for chunk in chunks:
    # We add a try-except block to handle any potential errors with a specific chunk
    try:
        chunk_summary = summarizer(chunk, max_length=130, min_length=25, do_sample=False)
        summaries.append(chunk_summary[0]['summary_text'])
    except IndexError:
        # If a chunk fails for some reason, we'll just skip it
        print("A chunk could not be summarized and was skipped.")
        continue

# Step 3: Combine the summaries into a final, comprehensive summary
final_summary = " ".join(summaries)

# Step 4: Print the original and final combined summary
print("--- Original Text ---")
print(long_text)
print("\n--- Final Summary ---")
print(final_summary)