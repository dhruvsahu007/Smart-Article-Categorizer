import pandas as pd
import random
from typing import List, Tuple

class SampleDataGenerator:
    def __init__(self):
        self.categories = ["Tech", "Finance", "Healthcare", "Sports", "Politics", "Entertainment"]
        
        # Sample articles for each category
        self.sample_articles = {
            "Tech": [
                "Apple unveiled its latest iPhone with revolutionary AI capabilities and enhanced camera technology. The new device features a more powerful processor and improved battery life.",
                "Microsoft announced a major update to its Azure cloud platform, introducing new machine learning tools and enhanced security features for enterprise customers.",
                "Google's latest AI model demonstrates breakthrough performance in natural language understanding and generation, outperforming previous state-of-the-art systems.",
                "Tesla's new electric vehicle features advanced autopilot capabilities and can travel over 400 miles on a single charge with its innovative battery technology.",
                "Meta's virtual reality headset offers immersive experiences with high-resolution displays and advanced hand tracking technology for gaming and productivity.",
                "Amazon Web Services launches new quantum computing service, making quantum algorithms accessible to developers and researchers worldwide.",
                "Netflix develops new recommendation algorithm using deep learning to personalize content suggestions based on viewing patterns and preferences.",
                "Spotify integrates AI-powered music discovery features that analyze user behavior and create personalized playlists with enhanced accuracy.",
                "Intel releases new processor architecture optimized for artificial intelligence workloads and machine learning applications in data centers.",
                "Nvidia's latest graphics cards deliver unprecedented performance for gaming and professional applications with ray tracing capabilities."
            ],
            "Finance": [
                "The Federal Reserve announced a 0.25% interest rate increase, citing concerns about inflation and economic growth in the coming quarter.",
                "Bitcoin reached a new all-time high this week, driven by institutional investment and growing acceptance of cryptocurrency as a legitimate asset class.",
                "Major banks reported strong quarterly earnings, with increased lending activity and improved credit quality contributing to profit growth.",
                "The stock market experienced significant volatility following concerns about global trade tensions and their potential impact on economic growth.",
                "Gold prices surged to multi-year highs as investors sought safe-haven assets amid economic uncertainty and geopolitical tensions.",
                "Cryptocurrency regulations are being considered by lawmakers, with potential implications for digital asset trading and investment strategies.",
                "Real estate markets show signs of cooling as mortgage rates increase and affordability concerns affect buyer demand in major cities.",
                "Corporate earnings season reveals mixed results, with technology companies outperforming traditional sectors in revenue and profit growth.",
                "International trade agreements face challenges as countries negotiate new terms and address concerns about global supply chain disruptions.",
                "Investment firms are increasing their focus on ESG (Environmental, Social, and Governance) criteria when making portfolio decisions."
            ],
            "Healthcare": [
                "Breakthrough gene therapy treatment shows promising results for rare genetic disorders, offering hope for patients with previously untreatable conditions.",
                "New COVID-19 variant detected by health officials prompts updated vaccination recommendations and enhanced monitoring protocols.",
                "Telemedicine adoption continues to grow as patients and healthcare providers embrace remote consultation options for routine care.",
                "Clinical trials for Alzheimer's disease treatment show significant cognitive improvement in early-stage patients using novel therapeutic approaches.",
                "Artificial intelligence assists radiologists in detecting early-stage cancer with higher accuracy than traditional screening methods.",
                "Mental health awareness increases as healthcare systems implement comprehensive programs to address depression and anxiety disorders.",
                "Robotic surgery technology advances with new minimally invasive procedures that reduce recovery time and improve patient outcomes.",
                "Pharmaceutical companies collaborate on developing personalized medicine approaches based on individual genetic profiles and biomarkers.",
                "Healthcare data security becomes critical as hospitals invest in cybersecurity measures to protect patient information from breaches.",
                "Wearable health devices enable continuous monitoring of vital signs and early detection of health issues through advanced sensor technology."
            ],
            "Sports": [
                "The championship game ended in a thrilling overtime victory with the winning team securing their first title in over a decade.",
                "Olympic preparations are underway as athletes compete in qualifying events and training intensifies for the upcoming summer games.",
                "Professional basketball season begins with new player trades and coaching changes expected to shake up league standings.",
                "Soccer World Cup qualifiers continue with surprising upsets and dominant performances from traditional powerhouse teams.",
                "Tennis tournament features exciting matches as top-ranked players compete for prize money and ranking points in major competitions.",
                "Baseball season highlights include record-breaking performances and rookie players making significant impacts on team success.",
                "NFL draft analysis reveals strategic picks and potential game-changing players who could transform team dynamics and performance.",
                "Golf major championship produces unexpected winner as weather conditions and course difficulty challenge even the most experienced players.",
                "Ice hockey playoffs showcase intense competition with overtime games and outstanding goaltending performances throughout the tournament.",
                "Swimming records are broken at international competition as athletes push the limits of human performance with new training techniques."
            ],
            "Politics": [
                "Presidential campaign intensifies as candidates focus on key swing states and address critical issues affecting voter priorities.",
                "Congressional hearing examines proposed legislation on healthcare reform and its potential impact on American families and businesses.",
                "International summit addresses climate change policies and global cooperation efforts to reduce greenhouse gas emissions.",
                "Supreme Court decision on constitutional law has far-reaching implications for civil rights and individual liberties nationwide.",
                "Local elections demonstrate shifting political trends as voters express concerns about economic policies and social issues.",
                "Trade negotiations between major economies focus on tariff reductions and fair competition in global markets.",
                "Immigration policy debate continues as lawmakers consider comprehensive reform measures and border security enhancements.",
                "Voting rights legislation faces challenges as states implement new requirements and procedures for electoral participation.",
                "Foreign policy tensions escalate as diplomatic efforts seek to resolve conflicts and maintain international stability and peace.",
                "Government budget negotiations address spending priorities and deficit reduction measures while maintaining essential public services."
            ],
            "Entertainment": [
                "Hollywood blockbuster breaks box office records with spectacular visual effects and compelling storytelling that captivates audiences worldwide.",
                "Music festival lineup announced featuring top artists across multiple genres, promising an unforgettable experience for music lovers.",
                "Television series finale delivers emotional conclusion to beloved characters and storylines that have captivated viewers for years.",
                "Celebrity wedding creates media frenzy as fans celebrate the union of two popular stars in an elaborate ceremony.",
                "Award season approaches with critics and industry professionals speculating about potential winners in major categories.",
                "Streaming platform announces new original content including exclusive series and documentaries from acclaimed directors and producers.",
                "Broadway show revival features updated choreography and modern interpretations of classic musical theater productions.",
                "Video game release generates excitement among gamers with innovative gameplay mechanics and immersive storytelling experiences.",
                "Film festival showcases independent cinema and emerging filmmakers who challenge conventional narratives and artistic boundaries.",
                "Concert tour announcement sends fans into frenzy as tickets sell out within minutes for popular artist's highly anticipated performances."
            ]
        }
    
    def generate_dataset(self, samples_per_category: int = 10) -> pd.DataFrame:
        """Generate a balanced dataset with samples from each category."""
        data = []
        
        for category in self.categories:
            articles = self.sample_articles[category]
            
            # If we need more samples than available, use random selection with replacement
            if samples_per_category > len(articles):
                selected_articles = random.choices(articles, k=samples_per_category)
            else:
                selected_articles = random.sample(articles, samples_per_category)
            
            for article in selected_articles:
                data.append({
                    'text': article,
                    'category': category,
                    'category_id': self.categories.index(category)
                })
        
        df = pd.DataFrame(data)
        return df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    def get_category_mapping(self) -> dict:
        """Get mapping of category names to IDs."""
        return {cat: idx for idx, cat in enumerate(self.categories)}
    
    def save_dataset(self, filepath: str, samples_per_category: int = 10):
        """Generate and save dataset to file."""
        df = self.generate_dataset(samples_per_category)
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath} with {len(df)} samples")
        return df

if __name__ == "__main__":
    generator = SampleDataGenerator()
    dataset = generator.save_dataset("training_data.csv", samples_per_category=15)
    print(f"Generated dataset with {len(dataset)} samples")
    print(f"Categories: {dataset['category'].value_counts()}") 