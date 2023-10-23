import { Component } from '@angular/core';

@Component({
  selector: 'app-header',
  templateUrl: './header.component.html',
  styleUrls: ['./header.component.css']
})
export class HeaderComponent {


  selectedTab = 'home';
  constructor() { }

  ngOnInit() {
    this.setSelectedTabFromUrl();
  }

  /**
   * Selects a tab according to the given information in the url (see Tabs object for the allocation)
   */
  setSelectedTabFromUrl() {
    const url = window.location.href;
    console.log('1' + url);

    const routeStart = url.lastIndexOf('/') + 1;
    this.selectedTab = url.substr(routeStart);

    if (this.selectedTab === '') {
      this.selectedTab = 'cam';
    }
    console.log('2' + this.selectedTab);

  }

  setSelectedTab(tab: string) {
    this.selectedTab = tab;
  }

  goToGit() {
    window.location.href = 'https://github.com/ahmadshallouf/VersusQA';
  }



}
