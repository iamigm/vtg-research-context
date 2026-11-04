"""Shared browser actions for the directory-derived result hierarchy."""
import json

def select_view(page,entry_id):
 browser=page.locator('#view-browser')
 if browser.get_attribute('open') is None:browser.locator(':scope>summary').click()
 page.locator('#view-tree-items button[data-view='+json.dumps(entry_id)+']').click()
