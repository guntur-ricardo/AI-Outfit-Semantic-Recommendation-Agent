from datasets import load_dataset
import pandas as pd
import os


# pipenv run python scripts/generate_sample_dataset.py
def make_sample_csv(
        hf_dataset: str = "McAuley-Lab/Amazon-Reviews-2023",
        split: str = "full",
        category: str = "raw_meta_Amazon_Fashion",
        sample_size: int = 5000,
        output_path: str = "data/amazon_fashion_sample_enriched.csv",
):
    """
    Stream a small sample from the Hugging Face Fashion metadata split,
    extracting and flattening key fields for enrichment:
      - parent_asin -> product_id
      - title
      - description (list -> concatenated string)
      - features (list -> semicolon-separated) NOTES: Mainly material; IE: 100% cotton, elastic, rubber -- could be useful
      - categories (list -> ' > ' joined)
      - details (JSON string -> flattened 'key: value; ...') NOTES: This seems to be irrelevant to outfit semantic recommendation
      - average_rating
      - rating_number
      - price (convertable to float or None) NOTES: NOTES: This seems to be irrelevant to outfit semantic recommendation
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Stream the dataset
    ds_iter = load_dataset(
        hf_dataset,
        category,
        split=split,
        streaming=True,
        trust_remote_code=True
    )

    rows = []
    for example in ds_iter:
        pid = example.get("parent_asin")
        title = example.get("title") or ""
        if not pid or not title:
            continue

        # Flatten description list
        desc_list = example.get("description") or []
        if isinstance(desc_list, list):
            description = " ".join(desc_list)
        else:
            description = str(desc_list)

        # Flatten categories
        cats = example.get("categories") or []
        if isinstance(cats, list):
            categories = " > ".join(cats)
        else:
            categories = str(cats)

        # Ratings
        avg_rating = example.get("average_rating") or None

        rows.append({
            "product_id": pid,
            "title": title.strip(),
            "description": description.strip(),
            "categories": categories,
            "average_rating": avg_rating,
        })

        if len(rows) >= sample_size:
            break

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    make_sample_csv()

# Headers
# # ['main_category', 'title', 'average_rating', 'rating_number', 'features', 'description', 'price', 'images', 'videos', 'store', 'categories',
# 'details', 'parent_asin', 'bought_together', 'subtitle', 'author']

# Example data
# {
#   'main_category': 'AMAZON FASHION',
#   'title': "YUEDGE 5 Pairs Men's Moisture Control Cushioned Dry Fit Casual Athletic Crew Socks for Men (Blue, Size 9-12)",
#   'average_rating': 4.6,
#   'rating_number': 16,
#   'features': [],
#   'description': [],
#   'price': 'None',
#   'images': {
#     'hi_res': [
#       'https://m.media-amazon.com/images/I/81XlFXImFrS._AC_UL1500_.jpg',
#       'https://m.media-amazon.com/images/I/61+yVkHHQ3S._AC_UL1200_.jpg',
#       'https://m.media-amazon.com/images/I/61vbh6sLR1L._AC_UL1200_.jpg',
#       'https://m.media-amazon.com/images/I/71tRVQuan7S._AC_UL1500_.jpg',
#       'https://m.media-amazon.com/images/I/81BvTztKWGL._AC_UL1200_.jpg',
#       'https://m.media-amazon.com/images/I/71LytMHW9ML._AC_UL1200_.jpg',
#       'https://m.media-amazon.com/images/I/71wJKMbj5cS._AC_UL1500_.jpg'
#     ],
#     'large': [
#       'https://m.media-amazon.com/images/I/41+cCfaVOFS._AC_.jpg',
#       'https://m.media-amazon.com/images/I/41jBdP7etRS._AC_.jpg',
#       'https://m.media-amazon.com/images/I/41UGJiRe7UL._AC_.jpg',
#       'https://m.media-amazon.com/images/I/41zb4GR-lWS._AC_.jpg',
#       'https://m.media-amazon.com/images/I/612BT4t-uFL._AC_.jpg',
#       'https://m.media-amazon.com/images/I/51ExLGv3QwL._AC_.jpg',
#       'https://m.media-amazon.com/images/I/313iU0xDEkS._AC_.jpg'
#     ],
#     'thumb': [
#       'https://m.media-amazon.com/images/I/41+cCfaVOFS._AC_SR38,50_.jpg',
#       'https://m.media-amazon.com/images/I/41jBdP7etRS._AC_SR38,50_.jpg',
#       'https://m.media-amazon.com/images/I/41UGJiRe7UL._AC_SR38,50_.jpg',
#       'https://m.media-amazon.com/images/I/41zb4GR-lWS._AC_SR38,50_.jpg',
#       'https://m.media-amazon.com/images/I/612BT4t-uFL._AC_SR38,50_.jpg',
#       'https://m.media-amazon.com/images/I/51ExLGv3QwL._AC_SR38,50_.jpg',
#       'https://m.media-amazon.com/images/I/313iU0xDEkS._AC_SR38,50_.jpg'
#     ],
#     'variant': [
#       'MAIN',
#       'PT01',
#       'PT02',
#       'PT03',
#       'PT04',
#       'PT05',
#       'PT06'
#     ]
#   },
#   'videos': {
#     'title': [],
#     'url': [],
#     'user_id': []
#   },
#   'store': 'GiveGift',
#   'categories': [],
#   'details': '{"Package Dimensions": "10.31 x 8.5 x 1.73 inches; 14.82 Ounces", "Item model number": "DHES5PM21DH12", "Date First Available": "February 12, 2021"}',
#   'parent_asin': 'B08BHN9PK5',
#   'bought_together': None,
#   'subtitle': None,
#   'author': None
# }
