Read AGENT.md to understand the project.
Read app.py, test_app.py and schema.sql


Step1:
Add a new unit test file to cover the new score calculation implemented inside app.py function search_pages.
In particular:

- Add a unit test to verify back links weight more than pages without back link
- Add a unit test to cover use of variable LINK_BOOST_WEIGHT
- Add a unit test to cover use of variable LINK_BOOST_CAP
- Add a unit test to cover use of BM25_TITLE_WEIGHT

Step2: Do you see mistakes in the page rank implementation?


GG Answer:

  On Step 2: I do see one conceptual issue. The implementation is not really PageRank; it is an inbound-link-count boost on top of BM25. It also appears to count self-links as inbound
  links, since the SQL does not exclude from_page_id = to_page_id. I did not change that behavior because the spec only requested tests and a review note.