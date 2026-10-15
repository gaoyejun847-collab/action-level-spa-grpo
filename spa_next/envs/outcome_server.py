"""Original WebShop navigation without unused intermediate oracle scoring.

The legacy checkout enumerates every option combination on search/item pages.
V0 ignores those scores; retain its rendering/state transitions and inherit the
original purchase-time scoring unchanged. No legacy source files are modified.
"""
import json
import time

from webshop.web_agent_site.envs.web_agent_text_env import SimServer
from webshop.web_agent_site.engine.engine import (
    get_top_n_product_from_keywords, get_product_per_page, map_action_to_html,
)


class OutcomeServer(SimServer):
    def search_results(self, session_id, **kwargs):
        session = self.user_sessions[session_id]
        keywords = kwargs["keywords"]
        assert isinstance(keywords, list)
        page = kwargs.get("page", 1)
        session["page"], session["keywords"] = page, keywords
        session["actions"]["search"] += 1
        session["asin"], session["options"] = None, {}
        start = time.time()
        products = get_top_n_product_from_keywords(
            keywords, self.search_engine, self.all_products, self.product_item_dict)
        self.search_time += time.time() - start
        url = f"{self.base_url}/search_results/{session_id}/{'+'.join(keywords)}/{page}"
        start = time.time()
        html = map_action_to_html(
            "search", session_id=session_id, products=get_product_per_page(products, page),
            keywords=session["keywords"], page=page, total=len(products),
            instruction_text=session["goal"]["instruction_text"])
        self.render_time += time.time() - start
        return html, url

    def item_page(self, session_id, **kwargs):
        session = self.user_sessions[session_id]
        name = kwargs["clickable_name"]
        clickable = kwargs["text_to_clickable"][name]
        if clickable.get("class") is not None and clickable.get("class")[0] == "product-link":
            session["asin"] = name.upper()
            session["actions"]["asin"] += 1
            session["asins"].add(session["asin"])
        elif clickable.get("name") is not None:
            session["options"][clickable["name"].lower()] = name
            session["actions"]["options"] += 1
        product = self.product_item_dict[session["asin"]]
        url = (f"{self.base_url}/item_page/{session_id}/{session['asin']}/"
               f"{'+'.join(session['keywords'])}/{session['page']}/{json.dumps(session['options'])}")
        html = map_action_to_html(
            "click", session_id=session_id, product_info=product,
            keywords=session["keywords"], page=session["page"], asin=session["asin"],
            options=session["options"], instruction_text=session["goal"]["instruction_text"],
            show_attrs=self.show_attrs)
        return html, url
