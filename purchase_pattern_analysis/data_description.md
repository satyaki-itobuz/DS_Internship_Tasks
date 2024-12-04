# Data Description for Market Basket Analysis and Purchase Prediction

## **1. Orders Dataset**
Tracks each order made by users, providing insights into purchasing behavior over time.

| Column Name          | Description                                   |
|----------------------|-----------------------------------------------|
| `order_id`           | Unique identifier for each order.            |
| `user_id`            | Unique identifier for the user who placed the order. |
| `order_number`       | Sequential number of the order for each user. |
| `order_dow`          | Day of the week the order was placed (0–6).  |
| `order_hour_of_day`  | Hour of the day the order was placed (0–23). |
| `days_since_prior_order` | Days since the user’s last order.          |

---

## **2. Order Products Dataset**
Lists all products included in each order, helping us analyze co-purchases and product associations.

| Column Name          | Description                                   |
|----------------------|-----------------------------------------------|
| `order_id`           | Foreign key referencing `order_id` in Orders dataset. |
| `product_id`         | Unique identifier for each product.          |
| `add_to_cart_order`  | Sequence in which the product was added to the cart. |
| `reordered`          | Indicates if the product was reordered (1: Yes, 0: No). |

---

## **3. Products Dataset**
Contains details about products, enabling an understanding of product characteristics and relationships.

| Column Name          | Description                                   |
|----------------------|-----------------------------------------------|
| `product_id`         | Unique identifier for each product.          |
| `product_name`       | Name of the product.                         |
| `aisle_id`           | Identifier for the aisle where the product belongs. |
| `department_id`      | Identifier for the department where the product belongs. |

---
