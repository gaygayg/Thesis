import pandas as pd
# load the CSV file into a pandas DataFrame
#df = pd.read_csv('Training.csv')
# filter out rows containing either the sentence 'This is an ad' or the word 'promo' in the 'Tweet' column
unwanted_keywords = ['This is an ad', '#promo', '#selling adoptme','Selling #adoptme', 'discount', 'buy now', 'click here', 'buy now', 'buy me','game',
                     'my paypal is', 'handmade', '#handmade', 'for sale', 'selling content','#sellingcontent', '#Buyingcontent', 'lodibetlivestreaming',
                     'buying content', 'open for commission', '#secretseller', 'DM me', 'adoptme', 'adopt me', 'join fast', 'join', 'follow', 'socialselling',
                     '#Pakistan','Pakistan','#bundle','Bundles','bundles','#Bundles','#pakistan','Pre-orders','Pre-order', 'ayalaownercompany', '#imbrokehelpme',
                     '#pre-orders','DONATION','#donate','donations', 'win','Wins','#Win','#WINS','#pre-orders','#pre-orer', 'sellingrobloxaccount', 'support donating'
                     '#win','PRE-ORDER','#PRE-ORDER','selling a content','buying a content','buyingaccounts', 'sellingaccounts', 'askinghelp', 'president','#genshin',
                     'BuyingContent','SellingContent','#BuyingContent','#SellingContent','#Giveaways','giveaway','visit our', 'legitseller', 'manok','#genshinimpact',
                     'watch out','check it','Elliott', 'Thank God Freestyle', '#PayPal shares', 'did you know', '#staytuned', 'staytuned', 'ciosmanggatbagoong','bili na',
                     'follow for more', 'assignments', '#Payeer e-wallet:', '#review', 'wts/lfb', 'watch it', '#Commerzbank', 'calculate paypal fees', '#genshincommissions'
                     '#stockmarketcrash', '#stock', 'check this out', 'available now!', 'now available','New Phishing Kit Hijacks', 'dm us', 'sendgcash','pleasesend',
                     'out now', 'LEMONA TIN CAN ', 'bracelet', 'perfume', 'Mission 1 Crore','#BORNPINK', 'PayPal layoffs',  '#PayPal layoff', 'PayPal layoff:', 'boostselling',
                     'commission for', 'mod:', '@yahoo.com', 'jesus', '#jesus', '#jesuschrist', '#StandWithUkraine', '#pray', 'selling royale high', ' #BLACKPINK_WORLDTOUR',
                     'pray','#prayer', 'prayer', '#cancer', '#donate', '#fundraising ', '#exclusive','payment page', '#Stripe', 'roblox', 'livestream today', '#Chanyeol',           
                    'Purchase One ', 'Download the App:', 'get an instant', '#MWC2023Barcelona','CLOUD SERVER KPREA ', 'DM me if interested', 'sugar', 'selling roblox',
                    '#sugardaddy','#BoBFinance', 'Drop your', '@POTUS ','please sir send money','limited slots','debt',' twittersellingaccoung', 'Pwede po ba makahingi ng tulong,'                               
                    'My account number', 'account number','high quality','#webdesignuganda','#webdesign', 'CHEAP WEB HOSTING', 'onenetnews','buyampsell','#WORLDTOUR'
                     '#webdesigner','design','#E_Transfer', 'cosplay','Cashzine recently', '#ootd ','#CryptoNews','#SmallCap', '@CBinsights','#askingforhelp', 'What do you prefer?',    
                    '#gfm', '#Bitchute','#articles','#PayPal Money ','looking for a','cloud server','#cosplay', '#cosplaygirl','lodibetgamig', '#askinghelp', '#GForest update',                                           
                    'LIVE PODCAST:','#Web3Gaming','Art commission','Act Now','#Cardano','PayPal accepted','These #Loans Get #Cash', 'negosyocare','askinghelp' ,'retweet', '#retweetdonate'  
                    'The perfect Gifts','follow me','shitty system','Company Registration','#Web2','@VisionCryptoApp','@iProov','sponsor', 'insutech', 'which one would you recommend?', 
                    'Whatsapp/ Viber','Whatsapp:', 'Genesys','#india','Congratulations to deposit','#makemoneyonline','sellingtwitteraccounts', 'email id','#buying_content', 'if anyone could send me'
                     '#codecanyon','#vape','me some money','@DCEastSinghbhum ','dop','wts','comming soon','#Dubbz','Rent a #Website','lodibetpromotion', 'barcelona' ,'Nigeria', 'india' ,'RT DigitalEU'                                                                              
                    'RENT VPS EGYPT ', 'PayPal is having a minor', 'rent a','Four Asian','Price:','#PalUpNow!','sizes','full article' ,'latest news','#Gcash CEO is there in Barcelona! '  'pick up his girl'                                        
                    'sales!','users can now','Offer Easy Way','Crypto exchange','Crypto exchanges','#woke go','Gift Cards For Free', 'kucoinapp', 'get updated','#foodsample','Explained the use!', 'another step'
                    '#commissionsopen','#DigitalWallet for #NFTs','Email:','#zonauang','#makemoney','Join Monay', 'royalehightrades','tutorial','I can purchase that for you' ,'africans'  '#easkme'                                    
                    '#paypaluganda','earn easymoney','gift item crafted','Check here','#Tip an','#SB19','#LGBT','@GameStop', '#paypal $btc','web3', 'candleflame','#korea''#tokyo','#africa''Helping a brother out'
                     'click link download free','get free gift card amazon paypal','invitation code','try bigcash ', 'royalehigh','coinmarketmatch','get started','#CreditSuisse','Chins is pushing', 'China is pushing'
                     'buying genuine gemstone ','fundraiser', 'sign surveytime','comfortdelgro partner ','paypal allows', 'paypal daily giveaway ', 'send help', 'if anyone can donate', 'donate to'
                     ,'royalehighselling','paypal authorizes','pypl rallying ','think paypal order', 'goregistered ', 'pls help stranded ','http', 'sendhelp','PEOPLE WHO BELIEVE IN #CRYPTO:''Come test with us'
                     'pls help stranded ','started small ', 'anyone willing ', 'anyone able bless ', 'online survey', 'great opportunity ', 'help amazon ', 'please subscribe','http','EU RESEARCH','Get your #onlinebusiness'
                     'professional 4g','coronavirus','cbn allow ','paypal join ', 'download best app ', 'top 5 ', 'download app ', 'paypal holding inc ', 'weverse shopping ','Time flies','watercolor','#HeadsUp','big news'
                     'paypal launching','professional wifi ','sleighpalgiveaway ', 'commission open','art commission', 'paypal confirmed ', 'earn money ','#buying_content','#weeniesoftwitter','#pet','can donate'
                     'handmade','covid19', 'trading', 'advertise business', 'invested',' giveaway', 'ends soon ', 'follow me', 'bringing us closer','FB: ', 'IG: ', 'AVAIL','#indieartists','#pets', 'how to earn'
                    ,' #VALORANT','play','#Data #NFT','#NFT','#homeless','#PiNews','#PayPal + #merch','#nft''HOMELESS PERSON ' 'homeless','#friendsandfamily','stay tuned','shelter','#FedNow','which is better?''Another step towards'
                    ' #SportsBetting ', ' Sports Betting ',' #Sports ', ' HOMELESS PERSON', ' HOMELESS', 'order available ', 'molde of payment', '#Amazon','#SaudiArabia','Anyone for the love of god','my code is','thread below'
                      'PREMIUM ACCOUNT!''still available','Big things loading' ,'OneNETnews' '#LegalWeek2023:','#LegalWeek2023','#Legalweek',' #ShortFilm',' Initial coin offering','I thought there are still good people','if anyone could send me','#BreakingNews','use my code',' PM me for the price'
                     ,' kahit magkano lang po','Pwede po ba makahingi ng tulong','makahingi ng tulong','THE LINK IS RIGHT ABOVE', 'here is the link','#teenmom', 'PLEASE send me' 'use my code!' 'invite code','im selling'
                     , '#tokyo' '#korea' '#england', '#SouthAfrica','#microsoft'



                     ]

def filter_ads(tweets):
  
    filtered_tweets = []
    for tweet in tweets:
        tweet_content = tweet[1]
        # Check if the tweet content contains any of the unwanted keywords or phrases
        if any(keyword in tweet_content for keyword in unwanted_keywords):
            continue
        else:
            filtered_tweets.append(tweet)
    return filtered_tweets                                                               
# write the filtered data back to a CSV file
#df.to_csv('filtered_data12.csv', index=False)