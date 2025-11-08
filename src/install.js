(function(){
    const btn = document.getElementById('get-started');
    if(!btn) return;
    btn.addEventListener('click', function(){
        const url = 'https://www.instagram.com/';
        
        // Check if chrome.tabs API is available (extension context)
        if (typeof chrome !== 'undefined' && chrome.tabs) {
            // Get the current tab ID first
            chrome.tabs.query({ currentWindow: true, active: true }, function(currentTabs) {
                const currentTabId = currentTabs && currentTabs[0] ? currentTabs[0].id : null;
                
                // Create new tab and wait for it to load before closing current tab
                chrome.tabs.create({ url: url, active: true }, function(newTab) {
                    // Close the install tab after the new tab is created
                    if (currentTabId && currentTabId !== newTab.id) {
                        setTimeout(() => {
                            chrome.tabs.remove(currentTabId);
                        }, 100);
                    }
                });
            });
        } else {
            // Fallback for localhost/non-extension context
            const newTab = window.open(url, '_blank', 'noopener,noreferrer');
            if(!newTab){
                // Popup blocked fallback: create an anchor and click it
                const a = document.createElement('a');
                a.href = url;
                a.target = '_blank';
                a.rel = 'noopener noreferrer';
                a.style.display = 'none';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }

            // Try to close this tab
            setTimeout(()=>{
                try{
                    window.close();
                    if(!window.closed){
                        window.location.replace('about:blank');
                    }
                }catch(e){
                    window.location.replace('about:blank');
                }
            }, 200);
        }
    });
})();
